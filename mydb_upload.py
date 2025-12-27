import subprocess
import os
def restore_mysql():
    host = "../.."
    user = "../.."
    password = "../.."
    database = "../.."
    dump_file = "../.."
    env = os.environ.copy()
    env["MYSQL_PWD"] = password
    # Step 1: Create DB if not exists
    subprocess.run(
        ["mysql", f"-h{host}", f"-u{user}", "-e", f"CREATE DATABASE IF NOT EXISTS {database};"],
        env=env,
        check=True
    )
    # Step 2: Restore data
    cmd = ["mysql", f"-h{host}", f"-u{user}", database]
    with open(dump_file, "r") as f:
        subprocess.run(cmd, stdin=f, env=env, check=True)
    print(f":white_check_mark: Data restored to database '{database}' from {dump_file}")
if __name__ == "__main__":
    restore_mysql()


















