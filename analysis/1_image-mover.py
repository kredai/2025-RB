
import shutil
import pathlib

num = 4
run = "cancore2_scan-28"
cwd = "C:/Nexus/Academic/#OneDrive [IISc]/Coursework/Projects/2025-04 [Ramray Bhat]/CC3D/analysis/cancore2/"
dest = cwd+run+"/!images"

for rep in range(1, num+1):
    src = cwd+run+f"/.scandata/replicate-{rep:02d}"

    def image_mover(src, dest):
        src = pathlib.Path(src)
        dest = pathlib.Path(dest)
        images = sorted(src.glob("**/*.png"), key=lambda x: int(''.join(filter(str.isdigit, x.stem))))
        for idx, img in enumerate(images):
            mcs = int(''.join(filter(str.isdigit, img.stem))) % 10**4
            cellfield_dir = dest/f"CellField_{mcs:04d}"/f"replicate-{rep:02d}"
            cellfield_dir.mkdir(parents=True, exist_ok=True)
            parent_name = img.parent.parent.parent.name
            iteration = int(''.join(filter(str.isdigit, parent_name)))
            new_img_name = f"cancore2_{mcs:04d}_{iteration:03d}.png"
            shutil.copy2(img, cellfield_dir / new_img_name)
            
    image_mover(src, dest)
    temp_dat_src = pathlib.Path(src) / "!temp.dat"
    temp_dat_dest = pathlib.Path(cwd) / run / f"{run}-{rep:02d}.dat"
    shutil.copy2(temp_dat_src, temp_dat_dest)


