import subprocess

CODER_BASE = [
    'coder',
    '--url', 'https://computeshare-access.tail641d9d.ts.net',
    '--use-keyring=false',
    '--no-version-warning',
    'ssh',
    '--disable-autostart'
]

# Create data/incoming on jwj-01
c = CODER_BASE + ['owner/jwj-01', 'mkdir -p ~/omok_worker/data/incoming']
subprocess.run(c)

# Send /tmp/test_ailab.npz from ai-lab to jwj-01/omok_worker/data/incoming/ailab_test_01.npz
p1 = subprocess.Popen(CODER_BASE + ['owner/ai-lab', 'cat /tmp/test_ailab.npz'], stdout=subprocess.PIPE)
p2 = subprocess.Popen(CODER_BASE + ['owner/jwj-01', 'cat > ~/omok_worker/data/incoming/ailab_test_01.npz'], stdin=p1.stdout)
p1.stdout.close()
p2.communicate()

# Verify on jwj-01
c_check = CODER_BASE + ['owner/jwj-01', 'ls -lh ~/omok_worker/data/incoming/']
res = subprocess.run(c_check, capture_output=True, text=True)
print("jwj-01 incoming dir:\n", res.stdout)
