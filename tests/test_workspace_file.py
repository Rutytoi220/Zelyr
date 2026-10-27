"""Test suite for ManageWorkspaceFileTool operations, safety guards, and atomic rollback."""

import asyncio
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from zelyr.tools.workspace_file import ManageWorkspaceFileTool, get_last_modified_file


class TestWorkspaceFileOperations(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="zelyr_test_workspace_"))
        self.tool = ManageWorkspaceFileTool()

    async def asyncTearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    async def test_write_and_read_with_line_budget(self):
        """Test write on new file and read with max_lines truncation."""
        target_file = self.test_dir / "sample.txt"
        initial_content = "Line 1: Hello World\nLine 2: Target to replace\nLine 3: Final line\n"

        res_write = await self.tool.execute({
            "action": "write",
            "path": str(target_file),
            "content": initial_content,
        })
        self.assertTrue(res_write.success)
        self.assertTrue(target_file.is_file())
        self.assertEqual(target_file.read_text(encoding="utf-8"), initial_content)

        res_read = await self.tool.execute({
            "action": "read",
            "path": str(target_file),
            "max_lines": 2,
        })
        self.assertTrue(res_read.success)
        self.assertEqual(res_read.output.get("lines_read"), 2)
        self.assertEqual(res_read.output.get("total_lines"), 3)
        self.assertTrue(res_read.output.get("truncated"))

    async def test_write_overwriting_creates_backup(self):
        """Test overwriting existing file creates backup copy."""
        target_file = self.test_dir / "overwrite_test.txt"
        initial_content = "Original Content\n"
        await self.tool.execute({
            "action": "write",
            "path": str(target_file),
            "content": initial_content,
        })

        updated_content = "Overwritten line A\nOverwritten line B\n"
        res_write2 = await self.tool.execute({
            "action": "write",
            "path": str(target_file),
            "content": updated_content,
        })
        self.assertTrue(res_write2.success)
        backup_path = res_write2.output.get("backup")
        self.assertIsNotNone(backup_path)
        self.assertTrue(Path(backup_path).is_file())
        self.assertEqual(Path(backup_path).read_text(encoding="utf-8"), initial_content)

    async def test_patch_single_match_succeeds(self):
        """Test patching single occurrence replaces text and creates backup."""
        target_file = self.test_dir / "patch_test.txt"
        initial_content = "Line A\nTarget String\nLine C\n"
        await self.tool.execute({
            "action": "write",
            "path": str(target_file),
            "content": initial_content,
        })

        res_patch = await self.tool.execute({
            "action": "patch",
            "path": str(target_file),
            "search_text": "Target String",
            "replace_text": "Replaced String",
        })
        self.assertTrue(res_patch.success)
        self.assertEqual(target_file.read_text(encoding="utf-8"), "Line A\nReplaced String\nLine C\n")
        patch_backup = res_patch.output.get("backup")
        self.assertIsNotNone(patch_backup)
        self.assertEqual(Path(patch_backup).read_text(encoding="utf-8"), initial_content)

    async def test_patch_safety_guards(self):
        """Test 0 matches and >1 ambiguous matches abort without mutating."""
        target_file = self.test_dir / "guard_test.txt"
        initial_content = "Unique Line\n"
        await self.tool.execute({
            "action": "write",
            "path": str(target_file),
            "content": initial_content,
        })

        # 0 matches
        res_0 = await self.tool.execute({
            "action": "patch",
            "path": str(target_file),
            "search_text": "Non-existent",
            "replace_text": "Whatever",
        })
        self.assertFalse(res_0.success)
        self.assertIn("appeared 0 times", res_0.error)
        self.assertEqual(target_file.read_text(encoding="utf-8"), initial_content)

        # >1 matches
        multi_content = "duplicate\nmiddle\nduplicate\n"
        target_file.write_text(multi_content, encoding="utf-8")
        res_multi = await self.tool.execute({
            "action": "patch",
            "path": str(target_file),
            "search_text": "duplicate",
            "replace_text": "replacement",
        })
        self.assertFalse(res_multi.success)
        self.assertIn("appeared 2 times", res_multi.error)
        self.assertEqual(target_file.read_text(encoding="utf-8"), multi_content)

    async def test_list_tree_respects_gitignore(self):
        """Test list_tree filters gitignored files and default hidden directories."""
        (self.test_dir / "src").mkdir()
        (self.test_dir / "src" / "index.py").write_text("print(1)", encoding="utf-8")
        (self.test_dir / "node_modules").mkdir()
        (self.test_dir / "node_modules" / "pkg.js").write_text("dummy", encoding="utf-8")
        (self.test_dir / ".git").mkdir()
        (self.test_dir / ".git" / "config").write_text("git", encoding="utf-8")
        (self.test_dir / ".gitignore").write_text("secret.env\n*.tmp\n", encoding="utf-8")
        (self.test_dir / "secret.env").write_text("SECRET=123", encoding="utf-8")
        (self.test_dir / "temp.tmp").write_text("temp", encoding="utf-8")

        res_tree = await self.tool.execute({
            "action": "list_tree",
            "path": str(self.test_dir),
        })
        self.assertTrue(res_tree.success)
        entries = res_tree.output.get("entries", [])
        entry_names = [e["name"] for e in entries]

        self.assertIn("src", entry_names)
        self.assertIn(".gitignore", entry_names)
        self.assertNotIn("node_modules", entry_names)
        self.assertNotIn(".git", entry_names)
        self.assertNotIn("secret.env", entry_names)
        self.assertNotIn("temp.tmp", entry_names)


class TestWorkspaceUndo(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.tool = ManageWorkspaceFileTool()

    async def asyncTearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    async def test_write_patch_rollback_workflow(self):
        test_file = Path(self.test_dir) / "test_doc.txt"

        # 1. First write (initial file)
        res1 = await self.tool.execute({
            "action": "write",
            "path": str(test_file),
            "content": "Initial Line 1\nInitial Line 2\n",
        })
        self.assertTrue(res1.success)
        self.assertEqual(get_last_modified_file(), test_file)

        await asyncio.sleep(1.05)

        # 2. Second write (creates backup of initial content)
        res2 = await self.tool.execute({
            "action": "write",
            "path": str(test_file),
            "content": "Modified Line 1\nModified Line 2\n",
        })
        self.assertTrue(res2.success)
        self.assertIsNotNone(res2.output.get("backup"))
        self.assertEqual(test_file.read_text(encoding="utf-8"), "Modified Line 1\nModified Line 2\n")

        await asyncio.sleep(1.05)

        # 3. Patch operation (creates backup of modified content)
        res3 = await self.tool.execute({
            "action": "patch",
            "path": str(test_file),
            "search_text": "Modified Line 1",
            "replace_text": "Patched Line 1",
        })
        self.assertTrue(res3.success)
        self.assertEqual(test_file.read_text(encoding="utf-8"), "Patched Line 1\nModified Line 2\n")

        # 4. Rollback should restore the latest backup (which had 'Modified Line 1\nModified Line 2\n')
        res_undo = await self.tool.execute({
            "action": "rollback",
            "path": str(test_file),
        })
        self.assertTrue(res_undo.success)
        self.assertIn("restored_path", res_undo.output)
        self.assertEqual(test_file.read_text(encoding="utf-8"), "Modified Line 1\nModified Line 2\n")
        self.assertEqual(get_last_modified_file(), test_file)

    async def test_rollback_no_backup_error(self):
        unbacked_file = Path(self.test_dir) / "never_backed_up.xyz"
        unbacked_file.write_text("random content", encoding="utf-8")

        res = await self.tool.execute({
            "action": "rollback",
            "path": str(unbacked_file),
        })
        self.assertFalse(res.success)
        self.assertIn("No backup found", res.error)


if __name__ == "__main__":
    unittest.main()
