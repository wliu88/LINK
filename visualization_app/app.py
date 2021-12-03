from flask import Flask, render_template, url_for, request, redirect

import os
import pickle
import shutil


# 13
entity_property_types = ["class", "material", "color", "transparency", "dimension", "physical_property", "shape",
                         "temperature", "spatial_distribution", "dampness", "purity", "room", "specific_place"]
# 6 + 13
all_property_types = ["name", "price", "weight", "size", "state_description", "image"] + entity_property_types

# define paths
current_dir = os.path.abspath(__file__)
current_dir = "/".join(current_dir.split("/")[:-1])
data_dir = os.path.join(current_dir, "../data/LINK_dataset")

# move images to static folder so that the app can access them
data_img_dir = os.path.join(data_dir, "vis")
static_img_dir = "static/vis"
if not os.path.exists(static_img_dir):
    print("Copy images to {}...".format(static_img_dir))
    shutil.copytree(data_img_dir, static_img_dir)

# load objects
object_data_filename = os.path.join(data_dir, "object_data.pkl")
with open(object_data_filename, "rb") as fh:
    object_data, object_instance_data = pickle.load(fh)


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = data_dir


def query_data(condition_properties):

    candidate_object_instances = []
    for oiid in object_instance_data:
        matched_properties = list(condition_properties.keys())
        for p in list(condition_properties.keys()):
            if p in object_instance_data[oiid]:
                if type(object_instance_data[oiid][p]) == list:
                    if condition_properties[p] in object_instance_data[oiid][p]:
                        matched_properties.remove(p)
                        continue
                else:
                    if condition_properties[p] == object_instance_data[oiid][p]:
                        matched_properties.remove(p)
                        continue

            elif p in object_data[object_instance_data[oiid]["id"]]:
                if type(object_data[object_instance_data[oiid]["id"]][p]) == list:
                    if condition_properties[p] in object_data[object_instance_data[oiid]["id"]][p]:
                        matched_properties.remove(p)
                        continue
                else:
                    if condition_properties[p] == object_data[object_instance_data[oiid]["id"]][p]:
                        matched_properties.remove(p)
                        continue

        if len(matched_properties) == 0:
            candidate_object_instances.append(oiid)

    object_instances = []
    for oiid in candidate_object_instances:
        oid = object_instance_data[oiid]["id"]
        object_instance = {}
        for k in object_data[oid]:
            d = object_data[oid][k]
            if type(d) == list:
                d = ", ".join([str(x) for x in d])
            object_instance[k] = d
        for k in object_instance_data[oiid]:
            d = object_instance_data[oiid][k]
            if type(d) == list:
                d = ", ".join([str(x) for x in d])
            object_instance[k] = d
        object_instance["img_url"] = "/static/vis/{:04d}.png".format(oid)
        object_instances.append(object_instance)

    return object_instances


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        query = request.form['query']

        query_dict = {}
        for q in query.split(","):
            field, value = q.split("=")
            query_dict[field] = value

        objects = query_data(query_dict)

        return render_template('visualization_page.html', query_dict=query_dict, objects=objects)

    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)