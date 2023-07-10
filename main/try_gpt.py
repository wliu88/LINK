import openai
import json
import re
import ast

openai.api_key = "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"


def predict_property_with_gpt(query_role, given_d, query_values, verbose=False, num_retry=1):
    given_d_str = ", ".join(["{}:{}".format(p[0], p[1]) for p in given_d])
    query_values_str = ", ".join(list(query_values))
    query_str = "How likely is the object to have the following {} properties: {}.".format(query_role, query_values_str)
    question_template = "The robot detects an object with the following properties: {}.\n{}\nPlease provide a python dictionary mapping from each candidate property to a number between 0 and 1, indicating its likelihood."

    question = question_template.format(given_d_str, query_str)
    if verbose:
        print(question)

    answer_dict = None
    for _ in range(num_retry):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an assistant helping a robot to operate in homes"},
                    {"role": "user", "content": question},
                ],
                temperature=0,
                max_tokens=500,
                # default values below
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )

            if verbose:
                print(response)

            answer_str = response['choices'][0]['message']['content']
            print(answer_str)

            # extract python dictionary
            pattern = r"```python\n(.*?)```"
            matches = re.findall(pattern, answer_str, re.DOTALL)
            if len(matches) > 0:
                extracted_code = matches[0]
                answer_dict = eval(extracted_code[extracted_code.find("{"):])
            else:
                raise Exception("No Python code block found.")

        except Exception as e:
            print(e)

        if answer_dict is not None:
            break

    if verbose:
        print(answer_dict)

    return answer_dict


def test():
    given_d = [('color', 'clear'), ('dampness', 'wet'), ('dimension', 'big'), ('dimension', 'deep'), ('dimension', 'long'), ('dimension', 'thick'), ('material', 'glass'), ('physical_property', 'hard'), ('price', 'medium'), ('purity', 'normal'), ('room', 'dining_room'), ('shape', 'angular'), ('shape', 'curved'), ('shape', 'hollow'), ('size', 'medium'), ('spatial_distribution', 'half'), ('specific_place', 'on table'), ('temperature', 'cold'), ('transparency', 'transparent'), ('weight', 'medium')]
    query_role = "class"
    query_values = {'brush', 'bottle', 'sponge', 'spatula', 'bowl', 'can', 'cup', 'fork', 'ladle', 'pan', 'box'}
    answer_dict = predict_property_with_gpt(query_role, given_d, query_values, verbose=True, num_retry=1)


if __name__ == "__main__":
    test()

