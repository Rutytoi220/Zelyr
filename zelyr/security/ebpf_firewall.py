"""AI-Driven eBPF Kernel Firewall.

Uses BCC to hook into tcp_v4_connect/tcp_v6_connect to monitor outbound
network connections. Integrates with the EventBus to allow AI agents to
evaluate IP reputation and block malicious outbound traffic.
"""
import logging
import socket
import struct
import asyncio
from typing import Dict, Any, Optional

try:
    from bcc import BPF
    BCC_AVAILABLE = True
except ImportError:
    BCC_AVAILABLE = False

from zelyr.core.events import EventBus

logger = logging.getLogger(__name__)

# eBPF C code to hook TCP connect
BPF_PROGRAM = """
#include <uapi/linux/ptrace.h>
#include <net/sock.h>
#include <bcc/proto.h>

BPF_HASH(currsock, u32, struct sock *);

struct ipv4_data_t {
    u32 pid;
    u32 saddr;
    u32 daddr;
    u16 dport;
    char comm[TASK_COMM_LEN];
};
BPF_PERF_OUTPUT(ipv4_events);

int trace_connect_entry(struct pt_regs *ctx, struct sock *sk) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    // Stash the socket for the return probe
    currsock.update(&pid, &sk);
    return 0;
};

int trace_connect_return(struct pt_regs *ctx) {
    int ret = PT_REGS_RC(ctx);
    u32 pid = bpf_get_current_pid_tgid() >> 32;

    struct sock **skpp;
    skpp = currsock.lookup(&pid);
    if (skpp == 0) {
        return 0;   // missed entry
    }

    if (ret != 0) {
        // failed to connect
        currsock.delete(&pid);
        return 0;
    }

    struct sock *skp = *skpp;
    u16 family = 0;
    bpf_probe_read_kernel(&family, sizeof(family), &skp->__sk_common.skc_family);

    if (family == AF_INET) {
        struct ipv4_data_t data4 = {.pid = pid};
        bpf_probe_read_kernel(&data4.saddr, sizeof(u32), &skp->__sk_common.skc_rcv_saddr);
        bpf_probe_read_kernel(&data4.daddr, sizeof(u32), &skp->__sk_common.skc_daddr);
        bpf_probe_read_kernel(&data4.dport, sizeof(u16), &skp->__sk_common.skc_dport);
        bpf_get_current_comm(&data4.comm, sizeof(data4.comm));
        
        ipv4_events.perf_submit(ctx, &data4, sizeof(data4));
    }

    currsock.delete(&pid);
    return 0;
}
"""


class ZelyrEBPFFirewall:
    """Monitors and intercepts TCP connections using eBPF."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.bpf: Optional['BPF'] = None
        self._running = False
        self._loop = None
        
    def start(self):
        """Compiles and loads the eBPF program."""
        if not BCC_AVAILABLE:
            logger.warning("ZelyrEBPFFirewall: 'bcc' package not installed. Kernel Firewall disabled.")
            return

        try:
            self.bpf = BPF(text=BPF_PROGRAM)
            self.bpf.attach_kprobe(event="tcp_v4_connect", fn_name="trace_connect_entry")
            self.bpf.attach_kretprobe(event="tcp_v4_connect", fn_name="trace_connect_return")
            
            # Open perf buffer
            self.bpf["ipv4_events"].open_perf_buffer(self._handle_ipv4_event)
            
            self._running = True
            logger.info("ZelyrEBPFFirewall: Successfully attached kprobes to tcp_v4_connect.")
            
            # Start polling in background
            self._loop = asyncio.get_event_loop()
            self._loop.create_task(self._poll_perf_buffer())
            
        except Exception as e:
            logger.error(f"ZelyrEBPFFirewall: Failed to start eBPF - {e}")
            self._running = False

    def stop(self):
        """Detaches kprobes and cleans up."""
        self._running = False
        if self.bpf:
            try:
                self.bpf.cleanup()
                logger.info("ZelyrEBPFFirewall: Detached kprobes.")
            except Exception as e:
                logger.error(f"ZelyrEBPFFirewall: Error during cleanup - {e}")

    async def _poll_perf_buffer(self):
        """Continuously polls the BPF perf buffer."""
        while self._running:
            try:
                # bpf_poll with timeout
                self.bpf.perf_buffer_poll(timeout=100)
                await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"ZelyrEBPFFirewall: Error polling perf buffer - {e}")
                await asyncio.sleep(1)

    def _handle_ipv4_event(self, cpu, data, size):
        """Callback for when a TCP connection is established."""
        try:
            event = self.bpf["ipv4_events"].event(data)
            
            saddr = socket.inet_ntoa(struct.pack("I", event.saddr))
            daddr = socket.inet_ntoa(struct.pack("I", event.daddr))
            # dport is big endian
            dport = socket.ntohs(event.dport)
            comm = event.comm.decode('utf-8', 'replace')
            
            payload = {
                "pid": event.pid,
                "comm": comm,
                "saddr": saddr,
                "daddr": daddr,
                "dport": dport
            }
            
            logger.debug(f"eBPF Firewall: Intercepted TCP connect: {comm} (PID {event.pid}) -> {daddr}:{dport}")
            
            # Fire an event for the AI Auditor to evaluate
            self.event_bus.publish_sync("network.intercept.info", payload)
            
            # Simulate a block check (In production this is an async RPC to the SecurityAuditorAgent)
            # We mock the blocking in user-space for now by emitting the critical event if daddr is "evil"
            if daddr == "198.51.100.42": # Example malicious IP
                logger.warning(f"eBPF Firewall: MALICIOUS IP DETECTED! Emitting critical intercept event for PID {event.pid}")
                self.event_bus.publish_sync("network.intercept.critical", payload)
                # Production: forcefully kill the PID or use bpf_override_return
                import os
                import signal
                try:
                    os.kill(event.pid, signal.SIGKILL)
                    logger.info(f"eBPF Firewall: Forcefully killed malicious process {event.pid}")
                except Exception as e:
                    logger.error(f"eBPF Firewall: Failed to kill process {event.pid} - {e}")
                    
        except Exception as e:
            logger.error(f"ZelyrEBPFFirewall: Error handling event - {e}")

# Legacy AXIOM API compatibility aliases
AxiomEBPFFirewall = ZelyrEBPFFirewall
