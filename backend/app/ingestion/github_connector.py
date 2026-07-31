import os
import shutil
import subprocess
import tempfile
import uuid
from typing import Any
from app.cli.embed import SUPPORTED_EXTENSIONS, chunk_file_lines
from app.core.logging import logger
from app.embeddings.pipeline import EmbeddingItem

class GitHubConnector:
    """
    GitHub & Git Repository Connector for EchoMind.
    Clones remote git repositories or scans local git repositories, extracts Git commit metadata
    (commit hash, author, commit date, branch), tracks file paths and line numbers (start_line, end_line),
    and produces EmbeddingItems for the embedding pipeline.
    """
    def __init__(self, temp_dir_base: str = None) -> None:
        self.temp_dir_base = temp_dir_base or tempfile.gettempdir()

    def extract_git_metadata(self, repo_path: str) -> dict[str, str]:
        """Extracts current Git commit metadata using git command line tools."""
        metadata = {
            "commit_hash": "unknown",
            "author": "unknown",
            "commit_date": "unknown",
            "branch": "main"
        }

        if not os.path.exists(os.path.join(repo_path, ".git")):
            return metadata

        try:
            # Commit Hash
            res_hash = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if res_hash.returncode == 0:
                metadata["commit_hash"] = res_hash.stdout.strip()

            # Author and Date
            res_log = subprocess.run(
                ["git", "log", "-1", "--format=%an <%ae>|%cd"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if res_log.returncode == 0 and "|" in res_log.stdout:
                parts = res_log.stdout.strip().split("|")
                metadata["author"] = parts[0]
                metadata["commit_date"] = parts[1]

            # Branch
            res_branch = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if res_branch.returncode == 0 and res_branch.stdout.strip():
                metadata["branch"] = res_branch.stdout.strip()

        except Exception as e:
            logger.warning(f"Could not extract git metadata for '{repo_path}': {e}")

        return metadata

    def scan_repository(self, repo_path: str) -> list[EmbeddingItem]:
        """Scans a local repository directory, preserving line numbers and Git commit metadata."""
        items: list[EmbeddingItem] = []
        abs_repo_path = os.path.abspath(repo_path)

        if not os.path.exists(abs_repo_path):
            logger.error(f"Repository path '{abs_repo_path}' does not exist.")
            return items

        git_meta = self.extract_git_metadata(abs_repo_path)
        logger.info(
            f"Scanning repository at '{abs_repo_path}' (Branch: '{git_meta['branch']}', "
            f"Commit: '{git_meta['commit_hash'][:8]}')..."
        )

        ignored_dirs = {
            ".git", "node_modules", ".next", "__pycache__", ".venv", "venv",
            ".checkpoints", "outputs", "logs", ".idea", ".vscode"
        }

        files_to_process: list[str] = []
        for root, dirs, files in os.walk(abs_repo_path):
            dirs[:] = [d for d in dirs if d not in ignored_dirs]
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in SUPPORTED_EXTENSIONS or file in ["README.md", "docker-compose.yml", "Dockerfile"]:
                    files_to_process.append(os.path.join(root, file))

        for filepath in files_to_process:
            ext = os.path.splitext(filepath)[1].lower()
            filename = os.path.basename(filepath)
            rel_path = os.path.relpath(filepath, abs_repo_path)

            chunk_blocks = chunk_file_lines(filepath)
            for c in chunk_blocks:
                item_id = f"repo_{filename}_L{c['start_line']}_L{c['end_line']}_{uuid.uuid4().hex[:6]}"
                items.append(
                    EmbeddingItem(
                        id=item_id,
                        source="github",
                        content=c["content"],
                        meta_data={
                            "filepath": rel_path,
                            "filename": filename,
                            "extension": ext or "none",
                            "start_line": c["start_line"],
                            "end_line": c["end_line"],
                            "commit_hash": git_meta["commit_hash"],
                            "author": git_meta["author"],
                            "commit_date": git_meta["commit_date"],
                            "branch": git_meta["branch"],
                        }
                    )
                )

        logger.info(
            f"Repository scan complete for '{abs_repo_path}': "
            f"{len(files_to_process)} files -> {len(items)} line-annotated EmbeddingItems."
        )
        return items

    def clone_and_scan(self, repo_url: str, branch: str = "main") -> list[EmbeddingItem]:
        """Clones a remote GitHub repository to a temporary directory and scans its files."""
        temp_dir = tempfile.mkdtemp(prefix="echomind_github_", dir=self.temp_dir_base)
        try:
            logger.info(f"Cloning remote repository '{repo_url}' (branch: '{branch}') into '{temp_dir}'...")
            res = subprocess.run(
                ["git", "clone", "--depth", "1", "-b", branch, repo_url, temp_dir],
                capture_output=True,
                text=True,
                timeout=60
            )
            if res.returncode != 0:
                logger.error(f"Failed to clone repository '{repo_url}': {res.stderr}")
                return []

            return self.scan_repository(temp_dir)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
