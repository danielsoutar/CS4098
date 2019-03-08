import argparse
import glob
import matplotlib.pyplot as plt
from feature_extraction_helpers import load_linestring, load_shape
from descartes import PolygonPatch

# 2016_12_17__0690.annotations
# name = "2016_12_17__0765(Ext).annotations"

parser = argparse.ArgumentParser()

parser.add_argument("--src", help="absolute path to folder containing .annotations files")
parser.add_argument("--dest", help="absolute path to folder for output images")
args = parser.parse_args()

# name = "SN189_3879_job5476.annotations"


def visualise_shapes(src, dest):
    for annotations_filename in sorted(glob.glob(src)):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        linestring = load_linestring(annotations_filename)
        shape = load_shape(annotations_filename)

        patch = PolygonPatch(shape, fc="red", alpha=0.5, zorder=2)
        ax.add_patch(patch)

        xs = [x for (x, y) in linestring.coords]
        ys = [y for (x, y) in linestring.coords]

        ax.plot(xs, ys)
        annotations_offset = -(len(".annotations"))
        annotations_filename = annotations_filename.split('/')[-1]

        output_filename = dest + annotations_filename[0:annotations_offset] + ".png"  # remove .annotations, add .png instead
        plt.savefig(output_filename, dpi=fig.dpi, bbox_inches='tight')

# End of file
