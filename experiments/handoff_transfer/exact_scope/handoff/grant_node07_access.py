"""Grant named POSIX ACLs on node07 locally; never follow directory symlinks."""
import ctypes
import json
import os
from pathlib import Path
import pwd
import stat
import sys

lib=ctypes.CDLL('libacl.so.1',use_errno=True)
for name,args,result in [('acl_get_file',[ctypes.c_char_p,ctypes.c_int],ctypes.c_void_p),
                         ('acl_from_text',[ctypes.c_char_p],ctypes.c_void_p),
                         ('acl_to_text',[ctypes.c_void_p,ctypes.c_void_p],ctypes.c_void_p),
                         ('acl_set_file',[ctypes.c_char_p,ctypes.c_int,ctypes.c_void_p],ctypes.c_int),
                         ('acl_calc_mask',[ctypes.POINTER(ctypes.c_void_p)],ctypes.c_int),
                         ('acl_free',[ctypes.c_void_p],ctypes.c_int)]:
    fn=getattr(lib,name);fn.argtypes=args;fn.restype=result
user='yashas.kotre';uid=pwd.getpwnam(user).pw_uid
root=Path('/scratch/aryama.murthy');ACCESS=0x8000;DEFAULT=0x4000

def text(path,kind):
    a=lib.acl_get_file(os.fsencode(path),kind)
    if not a:raise OSError(ctypes.get_errno(),str(path))
    p=lib.acl_to_text(a,None)
    if not p:raise OSError(ctypes.get_errno(),str(path))
    result=ctypes.string_at(p).decode();lib.acl_free(p);lib.acl_free(a)
    return result

def grant(path,kind,mode):
    old=text(path,kind)
    if not old.strip():
        def perm(bits):return ''.join(c if bits & b else '-' for b,c in ((4,'r'),(2,'w'),(1,'x')))
        old=f'user::{perm((mode>>6)&7)}\ngroup::{perm((mode>>3)&7)}\nother::{perm(mode&7)}\n'
    rows=[line.split('\t')[0] for line in old.splitlines() if line and not line.startswith((f'user:{uid}:',f'user:{user}:'))]
    permissions='rwx' if kind==DEFAULT or stat.S_ISDIR(mode) or mode&0o111 else 'rw-'
    if kind==DEFAULT:
        # stat group bits reflect the ACL mask, not the owning group's rights.
        # Preserve owning-user/group/other permissions from the access ACL.
        base=[line.split('\t')[0] for line in text(path,ACCESS).splitlines()
              if line.startswith(('user::','group::','other::'))]
        rows=[line for line in rows if not line.startswith(('user::','group::','other::'))]+base
    rows.append(f'user:{uid}:{permissions}')
    a=ctypes.c_void_p(lib.acl_from_text(('\n'.join(rows)+'\n').encode()))
    if not a.value:raise OSError(ctypes.get_errno(),str(path))
    if lib.acl_calc_mask(ctypes.byref(a)) or lib.acl_set_file(os.fsencode(path),kind,a):
        raise OSError(ctypes.get_errno(),str(path))
    lib.acl_free(a)
    actual=text(path,kind)
    assert f'user:{user}:{permissions}' in actual or f'user:{uid}:{permissions}' in actual,(path,actual)

counts={'files':0,'directories':0,'symlinks_skipped':0}
for parent,dirs,files in os.walk(root,followlinks=False,onerror=lambda e:(_ for _ in ()).throw(e)):
    path=Path(parent);mode=path.stat().st_mode
    if '--defaults-only' not in sys.argv:grant(path,ACCESS,mode)
    grant(path,DEFAULT,mode);counts['directories']+=1
    for name in files:
        p=path/name
        if p.is_symlink():counts['symlinks_skipped']+=1;continue
        if '--defaults-only' not in sys.argv:grant(p,ACCESS,p.stat().st_mode)
        counts['files']+=1
    if counts['directories']%5000==0:print(json.dumps(counts),flush=True)
result=dict(user=user,uid=uid,root=str(root),**counts,
            root_access=text(root,ACCESS),root_default=text(root,DEFAULT),status='verified')
(root/'yashas-handoff-20260912/acl-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
