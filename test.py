import os
os.environ = {k: v.replace('\\','\\\\') for k,v in os.environ.items()}
