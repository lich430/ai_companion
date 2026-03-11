#!/usr/bin/env python3
"""Deploy wechat.py to Debian server via SSH/SCP"""
import paramiko
import os
from scp import SCPClient

HOST = "192.168.250.240"
USER = "root"
PASSWORD = "admin"
REMOTE_DIR = "/opt/wechat_auto"

LOCAL_FILES = [
    r"C:\Users\Administrator\Desktop\ai_companion\wechat.py",
    r"C:\Users\Administrator\Desktop\ai_companion\start_wechat.sh",
]

# Check .env
env_path = r"C:\Users\Administrator\Desktop\ai_companion\.env"
if os.path.exists(env_path):
    LOCAL_FILES.append(env_path)

def create_ssh_client():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=30)
    return client

def main():
    print(f"Connecting to {HOST}...")
    ssh = create_ssh_client()
    print("Connected!")

    # Create remote directory
    stdin, stdout, stderr = ssh.exec_command(f"mkdir -p {REMOTE_DIR}")
    stdout.read()
    print(f"Created {REMOTE_DIR}")

    # Upload files
    with SCPClient(ssh.get_transport()) as scp:
        for local_path in LOCAL_FILES:
            if os.path.exists(local_path):
                filename = os.path.basename(local_path)
                print(f"Uploading {filename}...")
                scp.put(local_path, REMOTE_DIR)
                print(f"  Done: {filename}")

    # Make script executable
    ssh.exec_command(f"chmod +x {REMOTE_DIR}/start_wechat.sh")
    print("Made start_wechat.sh executable")

    # Install dependencies
    print("Installing Python dependencies...")
    stdin, stdout, stderr = ssh.exec_command("pip3 install uiautomator2 requests python-dotenv loguru 2>&1")
    print(stdout.read().decode())

    # Create systemd service
    service_content = """[Unit]
Description=WeChat Auto Reply Service
After=network.target

[Service]
Type=simple
ExecStart=/bin/bash /opt/wechat_auto/start_wechat.sh
Restart=always
RestartSec=5
WorkingDirectory=/opt/wechat_auto

[Install]
WantedBy=multi-user.target
"""

    print("Creating systemd service...")
    stdin, stdout, stderr = ssh.exec_command(f"cat > /etc/systemd/system/wechat-auto.service << 'EOFSERVICE'\n{service_content}\nEOFSERVICE")
    stdout.read()

    ssh.exec_command("systemctl daemon-reload")
    ssh.exec_command("systemctl enable wechat-auto")
    ssh.exec_command("systemctl start wechat-auto")
    print("Service started!")

    # Check status
    stdin, stdout, stderr = ssh.exec_command("systemctl status wechat-auto --no-pager")
    print(stdout.read().decode())

    ssh.close()
    print("\nDeployment complete!")

if __name__ == "__main__":
    main()
