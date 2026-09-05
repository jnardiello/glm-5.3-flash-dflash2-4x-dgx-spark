#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap
import unittest


REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "scripts/model_manifest.py"
OLD_REV = "03eb5366286afd40d2221b1d9c63a6dd1ba4832e"
NEW_REV = "690b705278a3a58e538fcb37c2ca8b5f9511213c"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_tool(*args: str, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", str(TOOL), *args],
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )


class ManifestTests(unittest.TestCase):
    def test_real_releases_differ_only_in_chat_template(self) -> None:
        old = json.loads((REPO / f"scripts/node/model-manifests/{OLD_REV}.json").read_text())
        new = json.loads((REPO / f"scripts/node/model-manifests/{NEW_REV}.json").read_text())
        old_files = {item["path"]: (item["size"], item["sha256"]) for item in old["files"]}
        new_files = {item["path"]: (item["size"], item["sha256"]) for item in new["files"]}
        changed = [name for name in old_files if old_files[name] != new_files[name]]
        shards = [item for item in new["files"] if item["path"].startswith("model-")]
        self.assertEqual(changed, ["chat_template.jinja"])
        self.assertEqual(len(shards), 62)
        self.assertEqual(sum(item["size"] for item in shards), 328_337_455_672)
        fixture = REPO / "scripts/tests/fixtures/chat_template-690b705.jinja"
        self.assertEqual(digest(fixture.read_bytes()), new_files["chat_template.jinja"][1])

    def test_plan_detects_revision_delta_corruption_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files: list[dict] = []
            for number in range(1, 63):
                name = f"model-{number:05d}-of-00062.safetensors"
                data = f"shard-{number}".encode()
                (root / name).write_bytes(data)
                files.append({"path": name, "size": len(data), "sha256": digest(data)})
            old_template, new_template = b"old-template", b"new-template-longer"
            (root / "chat_template.jinja").write_bytes(old_template)
            files.append({"path": "chat_template.jinja", "size": len(new_template), "sha256": digest(new_template)})
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"schema": 1, "repository": "test/repo", "revision": "a" * 40, "files": files}))

            plan = run_tool("plan", str(manifest), str(root))
            self.assertEqual(plan.returncode, 0)
            self.assertEqual(plan.stdout.splitlines(), [f"SIZE\t{len(new_template)}\tchat_template.jinja"])
            required = sum(int(line.split("\t")[1]) for line in plan.stdout.splitlines())
            self.assertEqual(required, len(new_template))
            self.assertLess(required, sum(item["size"] for item in files))

            (root / "chat_template.jinja").write_bytes(new_template)
            self.assertEqual(run_tool("plan", str(manifest), str(root)).stdout, "")
            self.assertEqual(run_tool("verify", str(manifest), str(root)).returncode, 0)
            self.assertEqual(run_tool("plan", str(manifest), str(root)).stdout, "")

            (root / "model-00007-of-00062.safetensors").write_bytes(b"shard-X")
            corrupt = run_tool("plan", str(manifest), str(root))
            self.assertIn("SHA256\t7\tmodel-00007-of-00062.safetensors", corrupt.stdout)

    def test_required_file_cannot_be_excluded(self) -> None:
        manifest = REPO / f"scripts/node/model-manifests/{NEW_REV}.json"
        result = run_tool("exclude-check", str(manifest), "*.safetensors")
        self.assertEqual(result.returncode, 1)
        self.assertIn("required manifest files", result.stderr)


class FetchIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.checkout = self.root / "repo"
        (self.checkout / "scripts/lib").mkdir(parents=True)
        (self.checkout / "scripts/node/model-manifests").mkdir(parents=True)
        for relative in ("scripts/fetch-fp8-weights.sh", "scripts/model_manifest.py", "scripts/lib/common.sh"):
            source = REPO / relative
            target = self.checkout / relative
            shutil.copy2(source, target)
        self.new_rev, self.old_rev = "a" * 40, "b" * 40
        self.files: list[dict] = []
        for number in range(1, 63):
            name = f"model-{number:05d}-of-00062.safetensors"
            data = f"shard-{number}".encode()
            self.files.append({"path": name, "size": len(data), "sha256": digest(data)})
        self.new_template, self.old_template = b"new-template", b"old-template"
        self.files.append({"path": "chat_template.jinja", "size": len(self.new_template), "sha256": digest(self.new_template)})
        manifest = {"schema": 1, "repository": "test/repo", "revision": self.new_rev, "files": self.files}
        (self.checkout / f"scripts/node/model-manifests/{self.new_rev}.json").write_text(json.dumps(manifest))
        self.homes = self.root / "homes"
        self.hosts = ["head", "worker1", "worker2", "worker3"]
        for host in self.hosts:
            home = self.homes / host
            model = home / "model"
            model.mkdir(parents=True)
            for item in self.files[:-1]:
                (model / item["path"]).write_bytes(f"shard-{int(item['path'][6:11])}".encode())
            (model / "chat_template.jinja").write_bytes(self.new_template if host == "head" else self.old_template)
            os.utime(model / "chat_template.jinja", (1_700_000_000, 1_700_000_000))
            (model / ".glm53-fp8-synced").write_text(self.old_rev)
            (home / "tp4/scripts").mkdir(parents=True)
            (home / "tp4/node/model-manifests").mkdir(parents=True)
            shutil.copy2(TOOL, home / "tp4/scripts/model_manifest.py")
            shutil.copy2(
                self.checkout / f"scripts/node/model-manifests/{self.new_rev}.json",
                home / f"tp4/node/model-manifests/{self.new_rev}.json",
            )
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self._write_mocks()
        self._write_env()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_env(self) -> None:
        values = f"""
NODES="{' '.join(self.hosts)}"
MGMT_IPS="192.0.2.1 192.0.2.2 192.0.2.3 192.0.2.4"
MASTER_IP=192.0.2.1
MASTER_PORT=29520
API_PORT=8000
FABRIC_TARGETS=("10.1.0.2 10.4.0.4" "10.1.0.1 10.2.0.3" "10.2.0.2 10.3.0.4" "10.3.0.3 10.4.0.1")
RELAY_DEST=worker2
IMAGE=test/image
CONTAINER=test
LAUNCHER=launch.sh
MODEL_DIR='$HOME/model'
MODEL_REPO=test/repo
MODEL_REV={self.new_rev}
DRAFT_DIR='$HOME/draft'
PATCH_FILE='$HOME/patch.py'
NCCL_DIR='$HOME/nccl'
CACHE_DIR='$HOME/cache'
SERVED_NAME=test
MAX_MODEL_LEN=1
MAX_NUM_SEQS=1
KV_CACHE_DTYPE=auto
BATCHED_TOKENS=1
BLOCK_SIZE=1
GPU_MEM_UTIL=1
SPEC_TOKENS=1
ASYNC_SCHEDULING=0
"""
        (self.checkout / "cluster.env").write_text(textwrap.dedent(values))

    def _write_mocks(self) -> None:
        ssh = self.bin / "ssh"
        ssh.write_text(textwrap.dedent("""\
            #!/usr/bin/env python3
            import os, subprocess, sys
            args=sys.argv[1:]; i=0
            while i < len(args) and args[i].startswith('-'):
                i += 2 if args[i] == '-o' else 1
            host=args[i]; command=' '.join(args[i+1:])
            env=os.environ.copy(); env['HOME']=str(__import__('pathlib').Path(env['MOCK_HOST_ROOT'])/host)
            if os.environ.get('FAIL_MARKER_HOST') == host and command.startswith('marker='):
                raise SystemExit(74)
            result=subprocess.run(['bash','-c',command], input=sys.stdin.buffer.read(), env=env)
            raise SystemExit(result.returncode)
        """))
        rsync = self.bin / "rsync"
        rsync.write_text(textwrap.dedent("""\
            #!/usr/bin/env python3
            import os, pathlib, shutil, sys
            args=sys.argv[1:]
            dest=args[-1]; source=pathlib.Path(args[-2]); host, relative=dest.split(':',1)
            if os.environ.get('FAIL_RSYNC_HOST') == host: raise SystemExit(23)
            target=pathlib.Path(os.environ['MOCK_HOST_ROOT'])/host/relative
            for line in sys.stdin:
                name=line.rstrip('\\n'); out=target/name; out.parent.mkdir(parents=True,exist_ok=True)
                if '--ignore-times' not in args and out.exists() and out.stat().st_size == (source/name).stat().st_size and out.stat().st_mtime == (source/name).stat().st_mtime: continue
                shutil.copy2(source/name,out)
        """))
        for executable in (ssh, rsync):
            executable.chmod(0o755)

    def _run_fetch(self, **extra: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(
            HOME=str(self.homes / "head"),
            PATH=f"{self.bin}:{env['PATH']}",
            MOCK_HOST_ROOT=str(self.homes),
            MODEL_DISK_MARGIN_BYTES="0",
        )
        env.update(extra)
        return subprocess.run(
            ["bash", str(self.checkout / "scripts/fetch-fp8-weights.sh")],
            cwd=self.checkout,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_interruption_never_relabels_and_retry_is_idempotent(self) -> None:
        failed = self._run_fetch(FAIL_RSYNC_HOST="worker1")
        self.assertNotEqual(failed.returncode, 0, failed.stdout + failed.stderr)
        for host in self.hosts:
            self.assertEqual((self.homes / host / "model/.glm53-fp8-synced").read_text(), self.old_rev)

        completed = self._run_fetch()
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        for host in self.hosts:
            model = self.homes / host / "model"
            self.assertEqual((model / ".glm53-fp8-synced").read_text().strip(), self.new_rev)
            self.assertEqual((model / "chat_template.jinja").read_bytes(), self.new_template)

        idempotent = self._run_fetch(FAIL_RSYNC_HOST="worker1")
        self.assertEqual(idempotent.returncode, 0, idempotent.stdout + idempotent.stderr)
        self.assertIn("data transfer skipped", idempotent.stdout)

    def test_corrupt_marked_shard_is_not_silently_repaired(self) -> None:
        shard = self.homes / "worker2/model/model-00007-of-00062.safetensors"
        shard.write_bytes(b"shard-X")
        result = self._run_fetch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing/corrupt shard", result.stderr)
        for host in self.hosts:
            self.assertEqual((self.homes / host / "model/.glm53-fp8-synced").read_text(), self.old_rev)

    def test_marker_failure_reports_partial_state_and_retry_repairs_it(self) -> None:
        failed = self._run_fetch(FAIL_MARKER_HOST="worker2")
        self.assertNotEqual(failed.returncode, 0, failed.stdout + failed.stderr)
        self.assertIn("markers may be partial", failed.stderr)
        self.assertIn("do not restart", failed.stderr)
        self.assertEqual((self.homes / "worker1/model/.glm53-fp8-synced").read_text().strip(), self.new_rev)
        for host in ("head", "worker2", "worker3"):
            self.assertEqual((self.homes / host / "model/.glm53-fp8-synced").read_text(), self.old_rev)

        completed = self._run_fetch()
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        for host in self.hosts:
            self.assertEqual(
                (self.homes / host / "model/.glm53-fp8-synced").read_text().strip(),
                self.new_rev,
            )

    def test_installed_head_missing_two_shards_requires_explicit_repair(self) -> None:
        for number in (7, 8):
            (self.homes / f"head/model/model-{number:05d}-of-00062.safetensors").unlink()
        result = self._run_fetch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("large implicit repair", result.stderr)
        for host in self.hosts:
            self.assertEqual((self.homes / host / "model/.glm53-fp8-synced").read_text(), self.old_rev)


if __name__ == "__main__":
    unittest.main()
