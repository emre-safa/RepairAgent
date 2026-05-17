import fcntl
import os
import shutil

EXPERIMENTS_LIST = "experimental_setups/experiments_list.txt"

with open(EXPERIMENTS_LIST, "a+") as expl:
    fcntl.flock(expl.fileno(), fcntl.LOCK_EX)
    try:
        expl.seek(0)
        exps = expl.read().splitlines()
        last_exp = int(exps[-1].split("_")[1]) if exps else 0

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
        except Exception:
            shutil.rmtree(new_dir, ignore_errors=True)
            raise

        expl.seek(0, os.SEEK_END)
        expl.write(new_exp + "\n")
        expl.flush()
        os.fsync(expl.fileno())
    finally:
        fcntl.flock(expl.fileno(), fcntl.LOCK_UN)

print(new_exp)
