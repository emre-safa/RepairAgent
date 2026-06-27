import fcntl
import os
import re
import shutil

EXPERIMENTS_LIST = "experimental_setups/experiments_list.txt"
EXP_RE = re.compile(r"^experiment_(\d+)$")

with open(EXPERIMENTS_LIST, "a+") as expl:
    fcntl.flock(expl.fileno(), fcntl.LOCK_EX)
    try:
        expl.seek(0)
        # Tolerate a malformed file (missing newlines, hand-edits): take the
        # max integer from anything matching experiment_<int>, ignore the rest.
        nums = [int(m.group(1)) for m in (EXP_RE.match(line) for line in expl.read().splitlines()) if m]
        last_exp = max(nums) if nums else 0

        new_exp = "experiment_{}".format(last_exp + 1)
        new_dir = os.path.join("experimental_setups", new_exp)

        try:
            os.mkdir(new_dir)
            os.mkdir(os.path.join(new_dir, "logs"))
            os.mkdir(os.path.join(new_dir, "responses"))
            os.mkdir(os.path.join(new_dir, "external_fixes"))
            os.mkdir(os.path.join(new_dir, "saved_contexts"))
            os.mkdir(os.path.join(new_dir, "mutations_history"))
            os.mkdir(os.path.join(new_dir, "plausible_patches"))
            os.mkdir(os.path.join(new_dir, "token_stats"))
        except Exception:
            shutil.rmtree(new_dir, ignore_errors=True)
            raise

        expl.seek(0, os.SEEK_END)
        # If the file didn't end with a newline (corrupted or hand-edited),
        # add one before appending so we don't extend the corruption.
        if expl.tell() > 0:
            expl.seek(expl.tell() - 1)
            if expl.read(1) != "\n":
                expl.write("\n")
        expl.write(new_exp + "\n")
        expl.flush()
        os.fsync(expl.fileno())
    finally:
        fcntl.flock(expl.fileno(), fcntl.LOCK_UN)

print(new_exp)
