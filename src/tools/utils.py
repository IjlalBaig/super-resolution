import sys
from zipfile import ZipFile
import tarfile
from pathlib import Path
import subprocess as sp
from tqdm import tqdm
import yaml


def read_yaml(fpath):
    with open(fpath, 'r') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

def collect_fpaths(dpath, suffixes):
    """
    Parameters
    ----------
    dpath:  str
        Directory to search for files
    ftypes: list
        Extension types to collect

    Returns
    -------
        List of file paths in directory dpath, with extensions listed in ftypes
    """
    if not Path(dpath).expanduser().is_dir():
        return []
    return [p.__str__() for p in Path(dpath).expanduser().glob("**/*") if p.name.endswith(tuple(suffixes))]

def extract_file(src, dst, filetype="zip"):
    if filetype == "zip":
        with ZipFile(src,'r') as zip:
            namelist = zip.namelist()
            total = len([f for f in namelist if not f.endswith("/")])
            cmd = ["unzip", "-nu", src, "-d", dst]
            use_shell = True # faster to extract zip
    elif filetype == "tar":
        with tarfile.open(src) as archive:
            total = sum(1 for member in archive if member.isreg())
            cmd = ["tar", "-xf", src, "-C", dst]
            use_shell = False # doesn't extract tar file error thrown
    Path(dst).mkdir(parents=True, exist_ok=True)
    try:
        with tqdm(leave=False, total=total, desc="extracting {}".format(Path(src).name)) as t:
            with sp.Popen(cmd, shell=use_shell, bufsize=1, universal_newlines=True, stdout=sp.PIPE, stderr=sp.PIPE) as process:
                for line in process.stdout:
                    t.update()
                    sys.stdout.flush()

                process.stdout.close()
                return_code = process.wait()
                if return_code != 0:
                    raise sp.CalledProcessError(return_code, cmd)

    except sp.CalledProcessError as e:
        sys.stderr.write(
            "common::run_command() : [ERROR]: output = {}, error code = {}\n".format(e.output, e.returncode))