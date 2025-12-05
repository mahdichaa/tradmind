import paramiko
import os
import time

# Configuration
HOST = "104.167.198.58"
USER = "root"
PASSWORD = "tm=1526388"
DOMAIN = "app.trademind.website"
BACKEND_DIR = "/var/www/tradvio-backend"
FRONTEND_DIR = "/var/www/tradvio-frontend"
DB_NAME = "tradvio_db"
DB_USER = "tradvio"
DB_PASS = "tradvio_secure_pass" 

# Local paths
LOCAL_BACKEND = os.getcwd()
LOCAL_FRONTEND = os.path.abspath(os.path.join(os.path.dirname(LOCAL_BACKEND), "tradvio-clone-frontend"))
LOCAL_DIST = os.path.join(LOCAL_FRONTEND, "dist")

def create_ssh_client():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD)
    return client

def run_command(client, command):
    print(f"Running: {command}")
    stdin, stdout, stderr = client.exec_command(command)
    exit_status = stdout.channel.recv_exit_status()
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if exit_status != 0:
        print(f"Error executing {command}: {err}")
    return out

def upload_files(sftp, local_path, remote_path):
    print(f"Uploading from {local_path} to {remote_path}")
    for root, dirs, files in os.walk(local_path):
        if 'venv' in root or '.git' in root or '__pycache__' in root or 'node_modules' in root or '.idea' in root:
            continue
            
        rel_path = os.path.relpath(root, local_path)
        if rel_path == ".":
            remote_root = remote_path
        else:
            remote_root = os.path.join(remote_path, rel_path).replace("\\", "/")
        
        try:
            sftp.stat(remote_root)
        except FileNotFoundError:
            try:
                sftp.mkdir(remote_root)
            except Exception as e:
                print(f"Error creating dir {remote_root}: {e}")
            
        for file in files:
            local_file = os.path.join(root, file)
            remote_file = os.path.join(remote_root, file).replace("\\", "/")
            try:
                sftp.put(local_file, remote_file)
            except Exception as e:
                print(f"Failed to upload {local_file}: {e}")

def main():
    try:
        client = create_ssh_client()
        sftp = client.open_sftp()
        
        print("Connected to server.")
        
        # 1. System Setup
        print("Installing system dependencies...")
        run_command(client, "apt-get update")
        run_command(client, "apt-get install -y python3 python3-pip python3-venv nodejs npm postgresql postgresql-contrib nginx")
        
        # 2. Database Setup
        print("Setting up database...")
        run_command(client, f"sudo -u postgres psql -c \"CREATE USER {DB_USER} WITH PASSWORD '{DB_PASS}';\"")
        run_command(client, f"sudo -u postgres psql -c \"CREATE DATABASE {DB_NAME} OWNER {DB_USER};\"")
        run_command(client, f"sudo -u postgres psql -c \"GRANT ALL PRIVILEGES ON DATABASE {DB_NAME} TO {DB_USER};\"")
        run_command(client, f"sudo -u postgres psql -d {DB_NAME} -c \"CREATE EXTENSION IF NOT EXISTS pgcrypto;\"")

        # 3. Backend Deployment
        print("Deploying backend...")
        run_command(client, f"mkdir -p {BACKEND_DIR}")
        upload_files(sftp, LOCAL_BACKEND, BACKEND_DIR)
        
        print("Setting up backend venv...")
        run_command(client, f"cd {BACKEND_DIR} && python3 -m venv venv")
        run_command(client, f"cd {BACKEND_DIR} && venv/bin/pip install -r requirements.txt")
        run_command(client, f"cd {BACKEND_DIR} && venv/bin/pip install gunicorn uvicorn")
        
        # Create .env
        env_content = ""
        if os.path.exists(".env"):
            with open(".env", "r", encoding='utf-8') as f:
                env_content = f.read()
        
        new_env = []
        for line in env_content.splitlines():
            if line.startswith("DATABASE_URL="):
                new_env.append(f"DATABASE_URL=postgresql+psycopg://{DB_USER}:{DB_PASS}@localhost/{DB_NAME}")
            elif line.startswith("DEBUG="):
                new_env.append("DEBUG=false")
            elif line.startswith("CORS_ORIGINS="):
                 new_env.append(f"CORS_ORIGINS=http://{DOMAIN},https://{DOMAIN}")
            else:
                new_env.append(line)
                
        if not any(l.startswith("CORS_ORIGINS=") for l in new_env):
            new_env.append(f"CORS_ORIGINS=http://{DOMAIN},https://{DOMAIN}")
            
        with sftp.file(f"{BACKEND_DIR}/.env", "w") as f:
            f.write("\n".join(new_env))
            
        # Systemd Service
        service_content = f"""[Unit]
Description=Tradvio Backend
After=network.target

[Service]
User=root
WorkingDirectory={BACKEND_DIR}
ExecStart={BACKEND_DIR}/venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
"""
        with sftp.file("/etc/systemd/system/tradvio.service", "w") as f:
            f.write(service_content)
            
        run_command(client, "systemctl daemon-reload")
        run_command(client, "systemctl enable tradvio")
        run_command(client, "systemctl restart tradvio")
        
        # 4. Frontend Deployment
        print("Deploying frontend...")
        run_command(client, f"mkdir -p {FRONTEND_DIR}")
        upload_files(sftp, LOCAL_DIST, FRONTEND_DIR)
        
        # 5. Nginx Setup
        print("Configuring Nginx...")
        nginx_config = f"""server {{
    listen 80;
    server_name {DOMAIN};

    root {FRONTEND_DIR};
    index index.html;

    location / {{
        try_files $uri $uri/ /index.html;
    }}

    location /api {{
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }}
}}
"""
        with sftp.file("/etc/nginx/sites-available/tradvio", "w") as f:
            f.write(nginx_config)
            
        run_command(client, "ln -sf /etc/nginx/sites-available/tradvio /etc/nginx/sites-enabled/")
        run_command(client, "rm -f /etc/nginx/sites-enabled/default")
        run_command(client, "nginx -t")
        run_command(client, "systemctl restart nginx")
        
        print("Deployment Complete!")
        client.close()
    except Exception as e:
        print(f"Deployment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
