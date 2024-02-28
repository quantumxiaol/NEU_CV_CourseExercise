import moxing as mox

# src 起点 dst 目的地    from obs

src_url = "obs://task3/lung.zip"
dst_url = "./lung.zip"

mox.file.copy_parallel(src_url, dst_url)