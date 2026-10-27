with open('zelyr/gui/widgets/update_dialog.py', 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "percent = int((downloaded / total_size) * 100)" in line:
        insert_idx = i + 3
        break

logic = """            # Attempt to fetch sha256sum.txt
            import os
            import hashlib
            expected_hash = None
            try:
                base_url = self.url.rsplit('/', 1)[0]
                hash_url = f"{base_url}/sha256sum.txt"
                req_hash = urllib.request.Request(hash_url, headers={'User-Agent': 'Zelyr-Updater'})
                with urllib.request.urlopen(req_hash, timeout=5) as response:
                    hash_content = response.read().decode('utf-8')
                    filename = self.url.split('/')[-1]
                    for line in hash_content.splitlines():
                        if filename in line:
                            expected_hash = line.split()[0].strip()
                            break
            except Exception as e:
                logger.warning(f"Could not fetch or parse sha256sum.txt: {e}")
                
            if expected_hash:
                sha256 = hashlib.sha256()
                with open('/tmp/zelyr_update.tar.gz', 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        sha256.update(chunk)
                actual_hash = sha256.hexdigest()
                if actual_hash != expected_hash:
                    os.remove('/tmp/zelyr_update.tar.gz')
                    class VerificationError(Exception): pass
                    raise VerificationError(f"Hash mismatch. Expected {expected_hash}, got {actual_hash}")
"""
lines.insert(insert_idx, logic)

# Now find the script string
for i, line in enumerate(lines):
    if "script = f\"\"\"#!/bin/bash" in line:
        start_idx = i
        break
for i in range(start_idx, len(lines)):
    if '"""' in lines[i] and i > start_idx:
        end_idx = i
        break

new_script = """            script = f\"\"\"#!/bin/bash
sleep 2

# 1. Verification
if [ -f "/tmp/zelyr_extracted/Zelyr" ]; then
    EXEC_NAME="Zelyr"
elif [ -f "/tmp/zelyr_extracted/main.py" ]; then
    EXEC_NAME="main.py"
else
    echo "Update Failed: Core executable not found." > /tmp/zelyr_ota_update.log
    "{current_dir}/Zelyr" &
    exit 1
fi

# 2. Safe Backup
if mv "{current_dir}" "{current_dir}.bak"; then
    # 3. Atomic Move
    if mv /tmp/zelyr_extracted "{current_dir}"; then
        chmod +x "{current_dir}/$EXEC_NAME"
        
        # 4. Health Check
        if "{current_dir}/$EXEC_NAME" --health-check; then
            # Health check passed, remove backup
            rm -rf "{current_dir}.bak"
            "{current_dir}/$EXEC_NAME" &
            exit 0
        else
            # 5. Rollback Trap on Health Check Failure
            echo "Update Failed: Health check crashed. Rolling back." >> /tmp/zelyr_ota_update.log
            rm -rf "{current_dir}"
            mv "{current_dir}.bak" "{current_dir}"
            "{current_dir}/$EXEC_NAME" &
            exit 1
        fi
    else
        # 4. Rollback Trap on Move Failure
        echo "Update Failed: Could not move new build. Rolling back." >> /tmp/zelyr_ota_update.log
        mv "{current_dir}.bak" "{current_dir}"
        "{current_dir}/$EXEC_NAME" &
        exit 1
    fi
else
    echo "Update Failed: Could not backup current directory." >> /tmp/zelyr_ota_update.log
    exit 1
fi
\"\"\"
"""
lines[start_idx:end_idx+1] = [new_script]

with open('zelyr/gui/widgets/update_dialog.py', 'w') as f:
    f.writelines(lines)
