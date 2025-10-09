import pefile

pe = pefile.PE(r"C:\Windows\System32\opengl32.dll")
machine = pe.FILE_HEADER.Machine

if machine == 0x8664:
    print("64-bit DLL")
elif machine == 0x014c:
    print("32-bit DLL")
else:
    print("Unknown architecture")
