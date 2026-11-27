"""Optional read-only Linux AT-SPI2 desktop accessibility reader using D-Bus (jeepney).

Provides strictly scoped, read-only inspection of an unambiguously targeted desktop
window and explicitly targeted non-sensitive widgets without injecting input,
focusing windows, or enumerating unrelated applications.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple

from zelyr.engine.hyprland_ipc import HyprlandIPCClient

BACKEND_NAME = "atspi2_dbus_jeepney"

MAX_INSPECTED_DESCENDANTS = 200
MAX_WALK_DEPTH = 10
MAX_RETURNED_TEXT_LEN = 2048
MAX_SELECTOR_LEN = 512
MAX_APPS_SCAN = 200
MAX_WINDOWS_PER_APP = 50
DEFAULT_DBUS_TIMEOUT = 3.0

# Exact Atspi.StateType bit positions from GObject Introspection / AT-SPI2 specification.
ATSPI_STATE_BITS: Dict[str, int] = {
    "active": 1,
    "checkable": 41,
    "checked": 4,
    "editable": 7,
    "enabled": 8,
    "expanded": 10,
    "focusable": 11,
    "focused": 12,
    "indeterminate": 32,
    "read_only": 43,
    "selectable": 22,
    "selected": 23,
    "sensitive": 24,
    "showing": 25,
    "visible": 30,
}

_CHECKABLE_ROLES = frozenset({
    "check box",
    "checkbox",
    "radio button",
    "toggle button",
    "check menu item",
    "radio menu item",
    "switch",
})

_EXPANDABLE_ROLES = frozenset({
    "combo box",
    "tree item",
    "expander",
    "disclosure triangle",
    "menu",
})

_SENSITIVE_ROLES = frozenset({
    "password text",
    "password-text",
    "password",
})

_SENSITIVE_FIELD_RE = re.compile(
    r"(?:^|[_\W])(?:password|passwd|passphrase|secret|api[_\s-]?key|apikey|"
    r"auth[_\s-]?token|access[_\s-]?token|refresh[_\s-]?token|credential|"
    r"private[_\s-]?key|bearer|cookie|localstorage|sessionstorage|ssn)(?:$|[_\W])",
    flags=re.IGNORECASE,
)

_ALLOWED_SELECTOR_KEYS = frozenset({
    "id",
    "role",
    "name",
    "description",
    "index",
})

_FORBIDDEN_SELECTOR_RE = re.compile(
    r"(?:\b(?:xpath|eval|javascript|script|document\.cookie|localstorage|sessionstorage)\s*[:=])",
    flags=re.IGNORECASE,
)

_SIMPLE_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


class AccessibilityBackendError(RuntimeError):
    """Base error for accessibility inspection."""


class AccessibilityBackendUnavailable(AccessibilityBackendError):
    """Raised when required D-Bus AT-SPI2 service or binding is unavailable."""


class AccessibilityAmbiguousTarget(AccessibilityBackendError):
    """Raised when window or widget selector matches more than one candidate."""


class AccessibilityInaccessibleTarget(AccessibilityBackendError):
    """Raised when target window/widget exists or is expected but exposes no accessible nodes."""


@dataclass(frozen=True)
class DesktopWidgetSelector:
    """Validated, structured widget selector scoped inside a single desktop window."""

    raw: str
    widget_id: Optional[str] = None
    id_substring: Optional[str] = None
    role: Optional[str] = None
    name: Optional[str] = None
    name_substring: Optional[str] = None
    description: Optional[str] = None
    description_substring: Optional[str] = None
    index: Optional[int] = None

    def matches(self, node: "AccessibleWidgetInfo") -> bool:
        if self.widget_id is not None:
            nid = (node.accessible_id or "").strip()
            if not nid or (nid != self.widget_id and not nid.endswith("." + self.widget_id)):
                return False
        if self.id_substring is not None:
            nid = (node.accessible_id or "").lower()
            if self.id_substring.lower() not in nid:
                return False
        if self.role is not None:
            if (node.role or "").strip().lower() != self.role.strip().lower():
                return False
        if self.name is not None:
            if (node.name or "") != self.name:
                return False
        if self.name_substring is not None:
            if self.name_substring not in (node.name or ""):
                return False
        if self.description is not None:
            if (node.description or "") != self.description:
                return False
        if self.description_substring is not None:
            if self.description_substring not in (node.description or ""):
                return False
        return True


@dataclass(frozen=True)
class AccessibleWidgetInfo:
    """Bounded, privacy-safe snapshot of a single matched AT-SPI2 widget."""

    role: str
    name: str
    description: str
    accessible_id: str
    text: Optional[str]
    has_text_interface: bool
    states: Dict[str, bool]
    sensitive_redacted: bool
    window_title: str
    window_pid: Optional[int]
    window_address: Optional[str] = None

    @property
    def visible_text(self) -> str:
        if self.sensitive_redacted:
            return ""
        if self.has_text_interface and self.text is not None:
            return self.text
        return self.name or self.description or ""


@dataclass(frozen=True)
class DesktopWidgetQueryResult:
    """Structured result of resolving one target window and one widget selector."""

    backend: str
    window_found: bool
    window_accessible: bool
    widget_found: bool
    widget_accessible: bool
    inspected_descendants: int
    truncated: bool
    window_title: Optional[str] = None
    window_pid: Optional[int] = None
    window_address: Optional[str] = None
    widget: Optional[AccessibleWidgetInfo] = None
    error: Optional[str] = None
    unsupported: bool = False
    ambiguous: bool = False
    inaccessible: bool = False


class DesktopAccessibilityReader(Protocol):
    """Protocol for read-only desktop accessibility inspection backends."""

    backend_name: str

    def is_available(self) -> Tuple[bool, str]:
        ...

    def query_widget(
        self,
        *,
        window_target: str,
        selector: DesktopWidgetSelector,
        window_pid: Optional[int] = None,
        window_title: Optional[str] = None,
        window_address: Optional[str] = None,
        compositor_checked: bool = False,
        compositor_window_exists: Optional[bool] = None,
    ) -> DesktopWidgetQueryResult:
        ...


def decode_atspi_states(state_words: Iterable[int]) -> Dict[str, bool]:
    """Decode AT-SPI2 GetState() 32-bit word pair into explicit boolean flags."""
    words = [int(w) & 0xFFFFFFFF for w in (state_words or ())]
    w0 = words[0] if len(words) > 0 else 0
    w1 = words[1] if len(words) > 1 else 0

    def _has_bit(bit_idx: int) -> bool:
        if bit_idx < 32:
            return bool(w0 & (1 << bit_idx))
        return bool(w1 & (1 << (bit_idx - 32)))

    decoded = {name: _has_bit(bit) for name, bit in ATSPI_STATE_BITS.items()}
    enabled_and_sens = decoded["enabled"] and decoded["sensitive"]
    decoded["enabled"] = enabled_and_sens
    decoded["disabled"] = not enabled_and_sens
    return decoded


def is_sensitive_widget(
    *,
    role: str = "",
    name: str = "",
    description: str = "",
    accessible_id: str = "",
    attributes: Optional[Dict[str, str]] = None,
    text_value: Optional[str] = None,
) -> bool:
    """Return True if an accessible node represents a password/secret/credential field."""
    role_l = (role or "").strip().lower()
    if role_l in _SENSITIVE_ROLES:
        return True

    attrs = attributes if isinstance(attributes, dict) else {}
    for k, v in attrs.items():
        kl = str(k).strip().lower()
        vl = str(v).strip().lower()
        if kl in ("input-purpose", "text-input-type", "purpose") and "password" in vl:
            return True
        if _SENSITIVE_FIELD_RE.search(kl) or _SENSITIVE_FIELD_RE.search(vl):
            return True

    for label in (accessible_id, name, description):
        if label and _SENSITIVE_FIELD_RE.search(label):
            return True

    if isinstance(text_value, str) and len(text_value) >= 3:
        if role_l in ("text", "entry", "editbar") and set(text_value) <= {"\u2022", "*"}:
            return True

    return False


def _split_top_level_clauses(spec: str) -> List[str]:
    """Split comma-delimited selector clauses outside single or double quotes."""
    parts: List[str] = []
    cur: List[str] = []
    in_quote: Optional[str] = None
    escape = False
    for ch in spec:
        if escape:
            cur.append(ch)
            escape = False
            continue
        if ch == "\\":
            cur.append(ch)
            escape = True
            continue
        if in_quote:
            if ch == in_quote:
                in_quote = None
            cur.append(ch)
            continue
        if ch in ("'", '"'):
            in_quote = ch
            cur.append(ch)
            continue
        if ch == ",":
            tok = "".join(cur).strip()
            if tok:
                parts.append(tok)
            cur = []
            continue
        cur.append(ch)
    if in_quote is not None:
        raise ValueError("Unclosed quote in desktop widget selector.")
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def parse_desktop_selector(raw_selector: Any) -> Tuple[Optional[DesktopWidgetSelector], Optional[str]]:
    """Validate and parse documented widget selectors (#id or key=val[,key2=val2])."""
    if raw_selector is None or not isinstance(raw_selector, str):
        return None, "Desktop widget verification requires an explicit non-empty 'selector'."
    sel = raw_selector.strip()
    if not sel:
        return None, "Desktop widget verification requires a non-empty 'selector'."
    if len(sel) > MAX_SELECTOR_LEN:
        return None, f"Desktop widget selector exceeds {MAX_SELECTOR_LEN} char limit."
    if "\x00" in sel or any(ord(ch) < 32 for ch in sel):
        return None, "Desktop widget selector contains illegal control characters."
    if _FORBIDDEN_SELECTOR_RE.search(sel):
        return None, "Desktop widget selector contains a forbidden script/cookie/storage directive."

    if sel.startswith("#"):
        ident = sel[1:].strip()
        if not ident or not _SIMPLE_ID_RE.match(ident):
            return None, (
                f"Malformed ID selector '{sel}': expected #<accessible_id> containing only "
                "letters, digits, '.', '_', or '-'."
            )
        return DesktopWidgetSelector(raw=sel, widget_id=ident), None

    try:
        clauses = _split_top_level_clauses(sel)
    except ValueError as exc:
        return None, str(exc)

    if not clauses:
        return None, "Empty desktop widget selector."

    fields: Dict[str, Any] = {}
    has_primary_predicate = False

    for clause in clauses:
        if "~=" in clause:
            k, v = clause.split("~=", 1)
            op = "~="
        elif "=" in clause:
            k, v = clause.split("=", 1)
            op = "="
        else:
            return None, (
                f"Unsupported desktop widget selector syntax '{clause}'. "
                "Use '#<id>' or documented predicates: id=, id~=, role=, name=, name~=, description=, description~=, index=."
            )

        key = k.strip().lower()
        val = v.strip()
        if len(val) >= 2 and ((val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'")):
            val = val[1:-1]

        if key not in _ALLOWED_SELECTOR_KEYS:
            return None, (
                f"Unsupported desktop selector predicate '{key}'. "
                "Allowed keys: id, role, name, description, index."
            )
        if not val:
            return None, f"Desktop selector predicate '{key}{op}' requires a non-empty value."

        if key == "index":
            if op != "=":
                return None, "Desktop selector 'index' requires '='."
            if "index" in fields:
                return None, "Duplicate 'index' predicate in desktop selector."
            try:
                idx = int(val)
                if idx < 0 or idx > 99:
                    return None, "Desktop selector 'index' must be between 0 and 99."
            except ValueError:
                return None, f"Invalid integer 'index={val}' in desktop selector."
            fields["index"] = idx
            continue

        if key == "role" and op != "=":
            return None, "Desktop selector 'role' requires exact '=' match."

        dest_key = (
            "widget_id"
            if (key == "id" and op == "=")
            else ("id_substring" if key == "id" else f"{key}{'_substring' if op == '~=' else ''}")
        )
        if dest_key in fields:
            return None, f"Duplicate predicate '{key}{op}' in desktop selector."
        fields[dest_key] = val
        has_primary_predicate = True

    if not has_primary_predicate:
        return None, "Desktop selector requires at least one widget predicate (id, role, name, or description)."

    return DesktopWidgetSelector(raw=sel, **fields), None


def split_window_and_widget_target(
    target: str,
    *,
    explicit_window: Optional[str] = None,
    explicit_selector: Optional[str] = None,
) -> Tuple[str, str]:
    """Resolve window target and widget selector from structured fields or 'window::selector' string."""
    raw_t = str(target or "").strip()
    win = str(explicit_window).strip() if explicit_window else ""
    sel = str(explicit_selector).strip() if explicit_selector else ""

    if "::" in raw_t and not (win and sel):
        left, right = raw_t.split("::", 1)
        if not win:
            win = left.strip()
        if not sel:
            sel = right.strip()
    elif not win:
        win = raw_t

    return win, sel


def match_hyprland_clients(clients: List[Dict[str, Any]], target: str) -> List[Dict[str, Any]]:
    """Return ALL Hyprland windows matching address, exact/partial title, or class."""
    tgt = str(target or "").strip()
    if not tgt:
        return []
    tgt_l = tgt.lower()
    if tgt_l.startswith("address:"):
        addr = tgt_l.removeprefix("address:").strip()
        return [c for c in clients if str(c.get("address") or "").lower() == addr]
    if tgt_l.startswith("0x"):
        return [c for c in clients if str(c.get("address") or "").lower() == tgt_l]

    exact_matches = [
        c
        for c in clients
        if str(c.get("title") or c.get("initialTitle") or "").strip().lower() == tgt_l
        or str(c.get("class") or c.get("initialClass") or "").strip().lower() == tgt_l
    ]
    if exact_matches:
        return exact_matches

    return [
        c
        for c in clients
        if tgt_l in str(c.get("title") or c.get("initialTitle") or "").lower()
        or tgt_l in str(c.get("class") or c.get("initialClass") or "").lower()
    ]


class AtspiDbusAccessibilityBackend:
    """Read-only AT-SPI2 accessibility backend communicating directly via jeepney D-Bus."""

    backend_name = BACKEND_NAME

    def __init__(self, timeout: float = DEFAULT_DBUS_TIMEOUT) -> None:
        self.timeout = max(0.2, float(timeout))

    @staticmethod
    def _import_jeepney() -> Tuple[Any, Any, Any]:
        try:
            from jeepney import DBusAddress, new_method_call  # type: ignore
            from jeepney.io.blocking import open_dbus_connection  # type: ignore

            return DBusAddress, new_method_call, open_dbus_connection
        except Exception as exc:
            raise AccessibilityBackendUnavailable(
                f"Linux AT-SPI2 Python D-Bus binding 'jeepney' is unavailable: {exc}"
            ) from exc

    def _resolve_bus_address(self) -> str:
        env_addr = os.environ.get("AT_SPI_BUS_ADDRESS", "").strip()
        if env_addr:
            return env_addr
        DBusAddress, new_method_call, open_dbus_connection = self._import_jeepney()
        sess_addr = os.environ.get("DBUS_SESSION_BUS_ADDRESS", "").strip()
        if not sess_addr:
            raise AccessibilityBackendUnavailable(
                "DBUS_SESSION_BUS_ADDRESS is not set; AT-SPI2 accessibility bus cannot be discovered."
            )
        try:
            with open_dbus_connection(bus="SESSION") as sess:
                msg = new_method_call(
                    DBusAddress("/org/a11y/bus", "org.a11y.Bus", "org.a11y.Bus"),
                    "GetAddress",
                )
                reply = sess.send_and_get_reply(msg, timeout=self.timeout)
                addr = str(reply.body[0] or "").strip() if reply and reply.body else ""
                if not addr:
                    raise AccessibilityBackendUnavailable("org.a11y.Bus.GetAddress returned empty bus address.")
                return addr
        except AccessibilityBackendUnavailable:
            raise
        except Exception as exc:
            raise AccessibilityBackendUnavailable(
                f"Failed to query AT-SPI2 bus via org.a11y.Bus: {exc}"
            ) from exc

    def is_available(self) -> Tuple[bool, str]:
        try:
            addr = self._resolve_bus_address()
            DBusAddress, new_method_call, open_dbus_connection = self._import_jeepney()
            with open_dbus_connection(bus=addr) as a11y:
                msg = new_method_call(
                    DBusAddress("/org/a11y/atspi/accessible/root", "org.a11y.atspi.Registry", "org.a11y.atspi.Accessible"),
                    "GetChildren",
                )
                a11y.send_and_get_reply(msg, timeout=self.timeout)
            return True, f"AT-SPI2 registry reachable via {addr}"
        except AccessibilityBackendUnavailable as exc:
            return False, str(exc)
        except Exception as exc:
            return False, f"AT-SPI2 registry unreachable: {exc}"

    def _call(self, conn: Any, new_method_call: Any, DBusAddress: Any, bus: str, path: str, iface: str, method: str, sig: str = "", args: Tuple[Any, ...] = ()) -> Any:
        msg = new_method_call(DBusAddress(path, bus, iface), method, sig, args)
        reply = conn.send_and_get_reply(msg, timeout=self.timeout)
        return reply.body

    def _get_prop(self, conn: Any, new_method_call: Any, DBusAddress: Any, bus: str, path: str, iface: str, prop: str) -> Any:
        body = self._call(
            conn,
            new_method_call,
            DBusAddress,
            bus,
            path,
            "org.freedesktop.DBus.Properties",
            "Get",
            "ss",
            (iface, prop),
        )
        val = body[0] if body else None
        if isinstance(val, tuple) and len(val) == 2:
            return val[1]
        return val

    def _get_optional_prop(self, conn: Any, new_method_call: Any, DBusAddress: Any, bus: str, path: str, iface: str, prop: str, default: Any = "") -> Any:
        try:
            return self._get_prop(conn, new_method_call, DBusAddress, bus, path, iface, prop)
        except Exception:
            return default

    def _read_widget_node(
        self,
        conn: Any,
        new_method_call: Any,
        DBusAddress: Any,
        *,
        bus: str,
        path: str,
        window_title: str,
        window_pid: Optional[int],
        window_address: Optional[str],
    ) -> AccessibleWidgetInfo:
        role = str(
            self._call(conn, new_method_call, DBusAddress, bus, path, "org.a11y.atspi.Accessible", "GetRoleName")[0] or ""
        )
        raw_name = str(
            self._get_optional_prop(conn, new_method_call, DBusAddress, bus, path, "org.a11y.atspi.Accessible", "Name", "") or ""
        )
        raw_desc = str(
            self._get_optional_prop(conn, new_method_call, DBusAddress, bus, path, "org.a11y.atspi.Accessible", "Description", "") or ""
        )
        raw_acc_id = str(
            self._get_optional_prop(conn, new_method_call, DBusAddress, bus, path, "org.a11y.atspi.Accessible", "AccessibleId", "") or ""
        )
        state_words = self._call(
            conn, new_method_call, DBusAddress, bus, path, "org.a11y.atspi.Accessible", "GetState"
        )[0]
        states = decode_atspi_states(state_words)

        try:
            raw_attrs = self._call(
                conn, new_method_call, DBusAddress, bus, path, "org.a11y.atspi.Accessible", "GetAttributes"
            )[0]
            attrs = dict(raw_attrs) if isinstance(raw_attrs, dict) else {}
        except Exception:
            attrs = {}

        ifaces = self._call(
            conn, new_method_call, DBusAddress, bus, path, "org.a11y.atspi.Accessible", "GetInterfaces"
        )[0] or ()
        has_text_iface = "org.a11y.atspi.Text" in ifaces

        # Privacy check BEFORE reading Text payload when role/name/attributes already signal a secret field.
        sensitive = is_sensitive_widget(
            role=role,
            name=raw_name,
            description=raw_desc,
            accessible_id=raw_acc_id,
            attributes=attrs,
        )

        text_val: Optional[str] = None
        if not sensitive and has_text_iface:
            try:
                char_count = int(
                    self._get_optional_prop(conn, new_method_call, DBusAddress, bus, path, "org.a11y.atspi.Text", "CharacterCount", 0)
                    or 0
                )
                if char_count > 0:
                    bounded = min(char_count, MAX_RETURNED_TEXT_LEN)
                    text_val = str(
                        self._call(conn, new_method_call, DBusAddress, bus, path, "org.a11y.atspi.Text", "GetText", "ii", (0, bounded))[0]
                        or ""
                    )[:MAX_RETURNED_TEXT_LEN]
                else:
                    text_val = ""
            except Exception as exc:
                raise AccessibilityInaccessibleTarget(
                    f"Widget matched '{raw_acc_id or raw_name}' is inaccessible while reading Text interface: {exc}"
                ) from exc

        if not sensitive and is_sensitive_widget(
            role=role,
            name=raw_name,
            description=raw_desc,
            accessible_id=raw_acc_id,
            attributes=attrs,
            text_value=text_val,
        ):
            sensitive = True
            text_val = None

        if sensitive:
            text_val = None
            safe_name = raw_name[:256] if not _SENSITIVE_FIELD_RE.search(raw_name) else "[redacted]"
            safe_desc = "[redacted]"
        else:
            safe_name = raw_name[:MAX_RETURNED_TEXT_LEN]
            safe_desc = raw_desc[:MAX_RETURNED_TEXT_LEN]

        return AccessibleWidgetInfo(
            role=role,
            name=safe_name,
            description=safe_desc,
            accessible_id=raw_acc_id[:256],
            text=text_val,
            has_text_interface=has_text_iface,
            states=states,
            sensitive_redacted=sensitive,
            window_title=window_title,
            window_pid=window_pid,
            window_address=window_address,
        )

    def _find_target_window_node(
        self,
        conn: Any,
        new_method_call: Any,
        DBusAddress: Any,
        *,
        window_target: str,
        window_pid: Optional[int],
        window_title: Optional[str],
    ) -> Tuple[Optional[Tuple[str, str, str, Optional[int]]], Optional[DesktopWidgetQueryResult]]:
        """Locate one unambiguous target window node via PID-first or top-level window matching."""
        apps = list(
            self._call(
                conn,
                new_method_call,
                DBusAddress,
                "org.a11y.atspi.Registry",
                "/org/a11y/atspi/accessible/root",
                "org.a11y.atspi.Accessible",
                "GetChildren",
            )[0]
            or ()
        )[:MAX_APPS_SCAN]

        if window_pid is not None:
            pid_app_refs: List[Tuple[str, str]] = []
            for bus_name, obj_path in apps:
                try:
                    peer_pid = int(
                        self._call(
                            conn,
                            new_method_call,
                            DBusAddress,
                            "org.freedesktop.DBus",
                            "/org/freedesktop/DBus",
                            "org.freedesktop.DBus",
                            "GetConnectionUnixProcessID",
                            "s",
                            (bus_name,),
                        )[0]
                    )
                except Exception:
                    continue
                if peer_pid == int(window_pid):
                    pid_app_refs.append((bus_name, obj_path))

            if not pid_app_refs:
                return None, DesktopWidgetQueryResult(
                    backend=self.backend_name,
                    window_found=True,
                    window_accessible=False,
                    widget_found=False,
                    widget_accessible=False,
                    inspected_descendants=0,
                    truncated=False,
                    window_title=window_title or window_target,
                    window_pid=window_pid,
                    inaccessible=True,
                    error=(
                        f"Target desktop window '{window_title or window_target}' (pid={window_pid}) exists, "
                        "but exposes no AT-SPI2 accessible application interface (window inaccessible)."
                    ),
                )

            candidate_wins: List[Tuple[str, str, str]] = []
            for app_bus, app_path in pid_app_refs:
                try:
                    wins = list(
                        self._call(conn, new_method_call, DBusAddress, app_bus, app_path, "org.a11y.atspi.Accessible", "GetChildren")[0]
                        or ()
                    )[:MAX_WINDOWS_PER_APP]
                except Exception:
                    wins = []
                for wb, wp in wins:
                    wname = str(
                        self._get_optional_prop(conn, new_method_call, DBusAddress, wb, wp, "org.a11y.atspi.Accessible", "Name", "")
                        or ""
                    )
                    candidate_wins.append((wb, wp, wname))

            if not candidate_wins:
                return None, DesktopWidgetQueryResult(
                    backend=self.backend_name,
                    window_found=True,
                    window_accessible=False,
                    widget_found=False,
                    widget_accessible=False,
                    inspected_descendants=0,
                    truncated=False,
                    window_title=window_title or window_target,
                    window_pid=window_pid,
                    inaccessible=True,
                    error=(
                        f"Application pid={window_pid} registered on AT-SPI2, "
                        "but exposes no accessible window frames."
                    ),
                )

            expected_t = (window_title or window_target).strip().lower()
            exact = [w for w in candidate_wins if w[2].strip().lower() == expected_t]
            matched_wins = exact or [w for w in candidate_wins if expected_t in w[2].lower()]
            if not matched_wins and len(candidate_wins) == 1:
                matched_wins = candidate_wins
            if len(matched_wins) > 1:
                return None, DesktopWidgetQueryResult(
                    backend=self.backend_name,
                    window_found=True,
                    window_accessible=False,
                    widget_found=False,
                    widget_accessible=False,
                    inspected_descendants=0,
                    truncated=False,
                    window_title=window_title or window_target,
                    window_pid=window_pid,
                    ambiguous=True,
                    error=f"Ambiguous target: {len(matched_wins)} accessible windows inside pid={window_pid} match '{window_title or window_target}'.",
                )
            if not matched_wins:
                return None, DesktopWidgetQueryResult(
                    backend=self.backend_name,
                    window_found=True,
                    window_accessible=False,
                    widget_found=False,
                    widget_accessible=False,
                    inspected_descendants=0,
                    truncated=False,
                    window_title=window_title or window_target,
                    window_pid=window_pid,
                    inaccessible=True,
                    error=f"No accessible window matching '{window_title or window_target}' inside pid={window_pid}.",
                )
            wb, wp, wname = matched_wins[0]
            return (wb, wp, wname or window_title or window_target, window_pid), None

        # Fallback (when Hyprland compositor wasn't queried): scan top-level windows only (1 level deep), zero descendant leakage.
        tgt_l = window_target.strip().lower()
        exact_top: List[Tuple[str, str, str, Optional[int]]] = []
        partial_top: List[Tuple[str, str, str, Optional[int]]] = []
        for app_bus, app_path in apps:
            try:
                wins = list(
                    self._call(conn, new_method_call, DBusAddress, app_bus, app_path, "org.a11y.atspi.Accessible", "GetChildren")[0]
                    or ()
                )[:MAX_WINDOWS_PER_APP]
            except Exception:
                continue
            app_pid: Optional[int] = None
            for wb, wp in wins:
                wname = str(
                    self._get_optional_prop(conn, new_method_call, DBusAddress, wb, wp, "org.a11y.atspi.Accessible", "Name", "")
                    or ""
                )
                wid = str(
                    self._get_optional_prop(conn, new_method_call, DBusAddress, wb, wp, "org.a11y.atspi.Accessible", "AccessibleId", "")
                    or ""
                )
                if not wname and not wid:
                    continue
                is_exact = wname.strip().lower() == tgt_l or wid.strip().lower() == tgt_l
                is_partial = tgt_l in wname.lower() or tgt_l in wid.lower()
                if is_exact or is_partial:
                    if app_pid is None:
                        try:
                            app_pid = int(
                                self._call(
                                    conn,
                                    new_method_call,
                                    DBusAddress,
                                    "org.freedesktop.DBus",
                                    "/org/freedesktop/DBus",
                                    "org.freedesktop.DBus",
                                    "GetConnectionUnixProcessID",
                                    "s",
                                    (app_bus,),
                                )[0]
                            )
                        except Exception:
                            app_pid = None
                    rec = (wb, wp, wname, app_pid)
                    if is_exact:
                        exact_top.append(rec)
                    else:
                        partial_top.append(rec)

        chosen = exact_top if exact_top else partial_top
        if len(chosen) > 1:
            return None, DesktopWidgetQueryResult(
                backend=self.backend_name,
                window_found=False,
                window_accessible=False,
                widget_found=False,
                widget_accessible=False,
                inspected_descendants=0,
                truncated=False,
                ambiguous=True,
                error=f"Ambiguous desktop target '{window_target}': {len(chosen)} accessible windows matched.",
            )
        if not chosen:
            return None, DesktopWidgetQueryResult(
                backend=self.backend_name,
                window_found=False,
                window_accessible=False,
                widget_found=False,
                widget_accessible=False,
                inspected_descendants=0,
                truncated=False,
                inaccessible=True,
                error=f"Target window '{window_target}' could not be located on AT-SPI2 registry.",
            )
        return chosen[0], None

    def query_widget(
        self,
        *,
        window_target: str,
        selector: DesktopWidgetSelector,
        window_pid: Optional[int] = None,
        window_title: Optional[str] = None,
        window_address: Optional[str] = None,
        compositor_checked: bool = False,
        compositor_window_exists: Optional[bool] = None,
    ) -> DesktopWidgetQueryResult:
        """Inspect one uniquely identified target window for a single matching widget."""
        if compositor_checked and compositor_window_exists is False:
            return DesktopWidgetQueryResult(
                backend=self.backend_name,
                window_found=False,
                window_accessible=False,
                widget_found=False,
                widget_accessible=False,
                inspected_descendants=0,
                truncated=False,
                error=f"Target desktop window '{window_target}' is not open on compositor.",
            )

        addr = self._resolve_bus_address()
        DBusAddress, new_method_call, open_dbus_connection = self._import_jeepney()
        try:
            with open_dbus_connection(bus=addr) as conn:
                target_win, err_res = self._find_target_window_node(
                    conn,
                    new_method_call,
                    DBusAddress,
                    window_target=window_target,
                    window_pid=window_pid,
                    window_title=window_title,
                )
                if err_res is not None:
                    return err_res
                assert target_win is not None
                wb, wp, resolved_title, resolved_pid = target_win

                visited = {(wb, wp)}
                queue: deque[Tuple[str, str, int]] = deque()
                try:
                    root_children = list(
                        self._call(conn, new_method_call, DBusAddress, wb, wp, "org.a11y.atspi.Accessible", "GetChildren")[0]
                        or ()
                    )
                except Exception as exc:
                    return DesktopWidgetQueryResult(
                        backend=self.backend_name,
                        window_found=True,
                        window_accessible=False,
                        widget_found=False,
                        widget_accessible=False,
                        inspected_descendants=0,
                        truncated=False,
                        window_title=resolved_title,
                        window_pid=resolved_pid,
                        window_address=window_address,
                        inaccessible=True,
                        error=f"Target window '{resolved_title}' children could not be inspected: {exc}",
                    )

                for cb, cp in root_children:
                    if (cb, cp) not in visited:
                        queue.append((cb, cp, 1))

                inspected = 0
                truncated = False
                matches: List[AccessibleWidgetInfo] = []

                while queue:
                    if inspected >= MAX_INSPECTED_DESCENDANTS:
                        truncated = True
                        break
                    nb, np, depth = queue.popleft()
                    if (nb, np) in visited:
                        continue
                    visited.add((nb, np))
                    inspected += 1

                    try:
                        info = self._read_widget_node(
                            conn,
                            new_method_call,
                            DBusAddress,
                            bus=nb,
                            path=np,
                            window_title=resolved_title,
                            window_pid=resolved_pid,
                            window_address=window_address,
                        )
                    except AccessibilityInaccessibleTarget as acc_exc:
                        return DesktopWidgetQueryResult(
                            backend=self.backend_name,
                            window_found=True,
                            window_accessible=True,
                            widget_found=True,
                            widget_accessible=False,
                            inspected_descendants=inspected,
                            truncated=truncated,
                            window_title=resolved_title,
                            window_pid=resolved_pid,
                            window_address=window_address,
                            inaccessible=True,
                            error=str(acc_exc),
                        )
                    except Exception:
                        info = None

                    if info is not None and selector.matches(info):
                        matches.append(info)
                        if selector.index is None and len(matches) >= 2:
                            return DesktopWidgetQueryResult(
                                backend=self.backend_name,
                                window_found=True,
                                window_accessible=True,
                                widget_found=True,
                                widget_accessible=True,
                                inspected_descendants=inspected,
                                truncated=truncated,
                                window_title=resolved_title,
                                window_pid=resolved_pid,
                                window_address=window_address,
                                ambiguous=True,
                                error=(
                                    f"Ambiguous widget selector '{selector.raw}': matched multiple widgets inside "
                                    f"window '{resolved_title}'. Specify a more specific selector or 'index=N'."
                                ),
                            )
                        if selector.index is not None and len(matches) > selector.index:
                            return DesktopWidgetQueryResult(
                                backend=self.backend_name,
                                window_found=True,
                                window_accessible=True,
                                widget_found=True,
                                widget_accessible=True,
                                inspected_descendants=inspected,
                                truncated=truncated,
                                window_title=resolved_title,
                                window_pid=resolved_pid,
                                window_address=window_address,
                                widget=matches[selector.index],
                            )

                    if depth < MAX_WALK_DEPTH and inspected < MAX_INSPECTED_DESCENDANTS:
                        try:
                            children = list(
                                self._call(conn, new_method_call, DBusAddress, nb, np, "org.a11y.atspi.Accessible", "GetChildren")[0]
                                or ()
                            )
                            for cb, cp in children:
                                if (cb, cp) not in visited:
                                    queue.append((cb, cp, depth + 1))
                        except Exception:
                            pass

                if not matches:
                    if truncated:
                        return DesktopWidgetQueryResult(
                            backend=self.backend_name,
                            window_found=True,
                            window_accessible=True,
                            widget_found=False,
                            widget_accessible=False,
                            inspected_descendants=inspected,
                            truncated=True,
                            window_title=resolved_title,
                            window_pid=resolved_pid,
                            window_address=window_address,
                            inaccessible=True,
                            error=(
                                f"Widget selector '{selector.raw}' not found within capped {MAX_INSPECTED_DESCENDANTS} "
                                f"descendants of '{resolved_title}'."
                            ),
                        )
                    return DesktopWidgetQueryResult(
                        backend=self.backend_name,
                        window_found=True,
                        window_accessible=True,
                        widget_found=False,
                        widget_accessible=True,
                        inspected_descendants=inspected,
                        truncated=False,
                        window_title=resolved_title,
                        window_pid=resolved_pid,
                        window_address=window_address,
                        error=f"Widget matching '{selector.raw}' was not found inside accessible window '{resolved_title}'.",
                    )

                if selector.index is not None:
                    return DesktopWidgetQueryResult(
                        backend=self.backend_name,
                        window_found=True,
                        window_accessible=True,
                        widget_found=False,
                        widget_accessible=True,
                        inspected_descendants=inspected,
                        truncated=truncated,
                        window_title=resolved_title,
                        window_pid=resolved_pid,
                        window_address=window_address,
                        error=(
                            f"Widget selector '{selector.raw}' index {selector.index} out of range "
                            f"(only {len(matches)} matching widget(s) inside '{resolved_title}')."
                        ),
                    )

                return DesktopWidgetQueryResult(
                    backend=self.backend_name,
                    window_found=True,
                    window_accessible=True,
                    widget_found=True,
                    widget_accessible=True,
                    inspected_descendants=inspected,
                    truncated=truncated,
                    window_title=resolved_title,
                    window_pid=resolved_pid,
                    window_address=window_address,
                    widget=matches[0],
                )
        except AccessibilityBackendUnavailable:
            raise
        except Exception as exc:
            raise AccessibilityBackendUnavailable(f"AT-SPI2 query failed: {exc}") from exc


async def query_desktop_widget_via_accessibility(
    *,
    window_target: str,
    selector: DesktopWidgetSelector,
    backend: Optional[DesktopAccessibilityReader] = None,
    hyprland_client: Optional[Any] = None,
    hyprland_socket_path: Optional[str] = None,
    timeout: float = DEFAULT_DBUS_TIMEOUT,
) -> DesktopWidgetQueryResult:
    """Resolve optional compositor target identity cleanly, then inspect ONE window over AT-SPI2."""
    eff_backend = backend or AtspiDbusAccessibilityBackend(timeout=timeout)
    avail, avail_reason = eff_backend.is_available()
    if not avail:
        return DesktopWidgetQueryResult(
            backend=getattr(eff_backend, "backend_name", BACKEND_NAME),
            window_found=False,
            window_accessible=False,
            widget_found=False,
            widget_accessible=False,
            inspected_descendants=0,
            truncated=False,
            unsupported=True,
            error=avail_reason,
        )

    hypr = hyprland_client if hyprland_client is not None else HyprlandIPCClient(socket_path=hyprland_socket_path)
    window_pid: Optional[int] = None
    window_title: Optional[str] = None
    window_address: Optional[str] = None
    compositor_checked = False
    compositor_exists: Optional[bool] = None

    can_conn = getattr(hypr, "can_connect", None)
    if callable(can_conn) and can_conn():
        try:
            clients = await hypr.get_clients()
            if isinstance(clients, list):
                compositor_checked = True
                matched = match_hyprland_clients(clients, window_target)
                if len(matched) > 1:
                    return DesktopWidgetQueryResult(
                        backend=getattr(eff_backend, "backend_name", BACKEND_NAME),
                        window_found=True,
                        window_accessible=False,
                        widget_found=False,
                        widget_accessible=False,
                        inspected_descendants=0,
                        truncated=False,
                        ambiguous=True,
                        error=(
                            f"Ambiguous desktop target '{window_target}': matched {len(matched)} open compositor windows."
                        ),
                    )
                if len(matched) == 1:
                    compositor_exists = True
                    m0 = matched[0]
                    window_pid = int(m0["pid"]) if m0.get("pid") is not None else None
                    window_title = str(m0.get("title") or m0.get("initialTitle") or "") or None
                    window_address = str(m0.get("address") or "") or None
                else:
                    compositor_exists = False
        except Exception:
            compositor_checked = False

    import asyncio

    try:
        return await asyncio.to_thread(
            eff_backend.query_widget,
            window_target=window_target,
            selector=selector,
            window_pid=window_pid,
            window_title=window_title,
            window_address=window_address,
            compositor_checked=compositor_checked,
            compositor_window_exists=compositor_exists,
        )
    except AccessibilityBackendUnavailable as exc:
        return DesktopWidgetQueryResult(
            backend=getattr(eff_backend, "backend_name", BACKEND_NAME),
            window_found=bool(compositor_exists),
            window_accessible=False,
            widget_found=False,
            widget_accessible=False,
            inspected_descendants=0,
            truncated=False,
            unsupported=True,
            error=str(exc),
        )


def is_widget_checkable(widget: AccessibleWidgetInfo) -> bool:
    return bool(
        widget.states.get("checkable")
        or widget.states.get("checked")
        or widget.role.strip().lower() in _CHECKABLE_ROLES
    )


__all__ = [
    "ATSPI_STATE_BITS",
    "AccessibilityAmbiguousTarget",
    "AccessibilityBackendError",
    "AccessibilityBackendUnavailable",
    "AccessibilityInaccessibleTarget",
    "AccessibleWidgetInfo",
    "AtspiDbusAccessibilityBackend",
    "BACKEND_NAME",
    "DesktopAccessibilityReader",
    "DesktopWidgetQueryResult",
    "DesktopWidgetSelector",
    "MAX_INSPECTED_DESCENDANTS",
    "MAX_RETURNED_TEXT_LEN",
    "MAX_SELECTOR_LEN",
    "MAX_WALK_DEPTH",
    "decode_atspi_states",
    "is_sensitive_widget",
    "is_widget_checkable",
    "match_hyprland_clients",
    "parse_desktop_selector",
    "query_desktop_widget_via_accessibility",
    "split_window_and_widget_target",
]
