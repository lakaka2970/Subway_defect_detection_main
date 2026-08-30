# -*- coding: utf-8 -*-
"""
AutoDL 远程执行助手。

用法（凭据走环境变量，不落盘）:
    export AUTOSSH_HOST=connect.bjb2.seetacloud.com
    export AUTOSSH_PORT=43567
    export AUTOSSH_USER=root
    export AUTOSSH_PASS='<密码>'          # 仅首次推公钥时需要
    export AUTOSSH_KEY=$HOME/.ssh/id_ed25519

    python scripts/remote_exec.py key                      # 推送公钥，建立免密
    python scripts/remote_exec.py exec "nvidia-smi"        # 执行远程命令
    python scripts/remote_exec.py exec "df -h" --timeout 60
    python scripts/remote_exec.py put  <本地> <远程>
    python scripts/remote_exec.py get  <远程> <本地>
"""
from __future__ import annotations

import os
import sys
import stat
import time

import paramiko

HOST = os.environ.get("AUTOSSH_HOST", "")
PORT = int(os.environ.get("AUTOSSH_PORT", "22"))
USER = os.environ.get("AUTOSSH_USER", "root")
PASS = os.environ.get("AUTOSSH_PASS", "")
KEY = os.environ.get("AUTOSSH_KEY", os.path.expanduser("~/.ssh/id_ed25519"))


def connect(use_pass: bool = False, timeout: int = 30):
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if use_pass:
        cli.connect(HOST, port=PORT, username=USER, password=PASS,
                    timeout=timeout, allow_agent=False, look_for_keys=False)
    else:
        pkey = None
        if KEY and os.path.exists(KEY):
            for loader in (paramiko.Ed25519Key, paramiko.RSAKey):
                try:
                    pkey = loader.from_private_key_file(KEY)
                    break
                except Exception:
                    continue
        cli.connect(HOST, port=PORT, username=USER, pkey=pkey,
                    timeout=timeout, allow_agent=False, look_for_keys=False)
    return cli


def cmd_push_key():
    pub = KEY + ".pub"
    if not os.path.exists(pub):
        print("找不到公钥 %s" % pub)
        return 1
    with open(pub, encoding="utf-8") as fp:
        pubdata = fp.read().strip()
    cli = connect(use_pass=True)
    sftp = cli.open_sftp()
    # 确保 .ssh 存在且权限正确
    stdin, out, err = cli.exec_command(
        "mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys "
        "&& chmod 600 ~/.ssh/authorized_keys")
    out.channel.recv_exit_status()
    try:
        with sftp.open("/root/.ssh/authorized_keys", "r") as fp:
            cur = fp.read().decode("utf-8", "ignore")
    except IOError:
        cur = ""
    if pubdata.split()[1] in cur:
        print("公钥已存在，跳过")
    else:
        with sftp.open("/root/.ssh/authorized_keys", "a") as fp:
            fp.write(("\n" if cur and not cur.endswith("\n") else "") + pubdata + "\n")
        print("公钥已写入 /root/.ssh/authorized_keys")
    sftp.close()
    cli.close()
    # 验证免密
    cli2 = connect()
    stdin, out, err = cli2.exec_command("echo OK $(hostname)")
    print("免密验证:", out.read().decode().strip() or err.read().decode().strip())
    cli2.close()
    return 0


def cmd_exec(remote_cmd: str, timeout: int = 600, stream: bool = True):
    cli = connect()
    chan = cli.get_transport().open_session()
    chan.settimeout(timeout)
    chan.exec_command(remote_cmd)
    t0 = time.time()
    while True:
        if chan.recv_ready():
            data = chan.recv(65536).decode("utf-8", "ignore")
            if not data:
                break
            sys.stdout.write(data)
            sys.stdout.flush()
        if chan.recv_stderr_ready():
            data = chan.recv_stderr(65536).decode("utf-8", "ignore")
            if not data:
                break
            sys.stderr.write(data)
            sys.stderr.flush()
        if chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready():
            break
        if time.time() - t0 > timeout:
            print("\n[超时 %ds]" % timeout, file=sys.stderr)
            break
        time.sleep(0.2)
    rc = chan.recv_exit_status()
    cli.close()
    return rc


def cmd_put(local: str, remote: str):
    cli = connect()
    sftp = cli.open_sftp()
    if os.path.isdir(local):
        _put_dir(sftp, local, remote)
    else:
        sftp.put(local, remote)
        print("已上传 %s -> %s" % (local, remote))
    sftp.close()
    cli.close()
    return 0


def _put_dir(sftp, local, remote):
    try:
        sftp.stat(remote)
    except IOError:
        sftp.mkdir(remote)
    n = 0
    for name in os.listdir(local):
        lp = os.path.join(local, name)
        rp = remote.rstrip("/") + "/" + name
        if os.path.isdir(lp):
            _put_dir(sftp, lp, rp)
        else:
            sftp.put(lp, rp)
            n += 1
    print("  %s -> %s (%d 文件)" % (local, remote, n))


def cmd_get(remote: str, local: str):
    cli = connect()
    sftp = cli.open_sftp()
    sftp.get(remote, local)
    print("已下载 %s -> %s" % (remote, local))
    sftp.close()
    cli.close()
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    mode = sys.argv[1]
    if mode == "key":
        sys.exit(cmd_push_key())
    elif mode == "exec":
        sys.exit(cmd_exec(sys.argv[2],
                          int(sys.argv[3]) if len(sys.argv) > 3 else 600))
    elif mode == "put":
        sys.exit(cmd_put(sys.argv[2], sys.argv[3]))
    elif mode == "get":
        sys.exit(cmd_get(sys.argv[2], sys.argv[3]))
    else:
        print(__doc__)
        sys.exit(1)
