import re
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, MultiPolygon
from descartes import PolygonPatch

# 2016_12_17__0690.annotations
# name = "2016_12_17__0765(Ext).annotations"

name = "2016_12_17__0690.annotations"
hierarchy = open("./Daniel line/" + name, "r").read()

p = re.compile("<*")

hierarchy_tags = p.split(hierarchy)

print(len(hierarchy_tags))

x_tags = []
y_tags = []

for tag in hierarchy_tags:
    if "X=" in tag and "Y=" in tag:
        x = tag.split("\"")[1]
        y = tag.split("\"")[3]
        x_tags.append(int(x))
        y_tags.append(int(y))

tags = []


# We have 4 points

# For each of 4 points:
# - is within core tumour? (core_tumour.contains(point)) // inner line
# - if no, then check: tumour_front.contains(point) or invasive_front.contains(point) // second or third lines
#

for i, j in zip(x_tags, y_tags):
    tags.append((i, j))

fig = plt.figure(1)

# 1
ax = fig.add_subplot(111)

poly = Polygon(tags)


def plot_line(ax, ob):
    x, y = ob.xy


def construct_shape(poly, distance, delta, update):
    difference = 0
    if type(poly.buffer(distance)) is MultiPolygon:
        difference = update(difference, delta)
        while type(poly.buffer(distance + difference)) is MultiPolygon:
            difference = update(difference, delta)
        new_poly = poly.buffer(distance + difference)
        return new_poly.buffer(-difference)
    else:
        return poly.buffer(distance)


def construct_shape_add_on_new_shape(poly, distance, delta, update):
    padding = update(0, delta)
    difference = 0
    if type(poly.buffer(distance)) is MultiPolygon:
        print("Boo!")
        shape = poly
        difference = update(difference, delta)
        while type(shape.buffer(distance + padding)) is MultiPolygon:
            shape = shape.buffer(distance + padding)
            difference = update(difference, delta)
        return shape.buffer(-difference)
    else:
        return poly.buffer(distance)

# Iterative increasing approach
dilated = construct_shape(poly, 500, 1, lambda x, y: x + y)
contracted = construct_shape(dilated, -1000, 1, lambda x, y: x - y)

print(type(contracted))
print(MultiPolygon)

if type(contracted) is MultiPolygon:
    print("Uh oh. Contracted for " + name + " invalid.")
    int_x = [i[0] for po in contracted for i in po.exterior.coords]
    int_y = [i[1] for po in contracted for i in po.exterior.coords]
else:
    int_x = [i[0] for i in contracted.exterior.coords]
    int_y = [i[1] for i in contracted.exterior.coords]

if type(dilated) is MultiPolygon:
    print("Uh oh. Dilated for " + name + " invalid.")
    ext_x = [i[0] for po in dilated for i in po.exterior.coords]
    ext_y = [i[1] for po in dilated for i in po.exterior.coords]
else:
    ext_x = [i[0] for i in dilated.exterior.coords]
    ext_y = [i[1] for i in dilated.exterior.coords]


plt.plot(ext_x, ext_y, color="r")
plt.plot(int_x, int_y, color="b")

patch = PolygonPatch(poly, fc="red", alpha=0.5, zorder=2)
ax.add_patch(patch)

# plt.plot(x_tags, y_tags)
plt.show()



# End of file
