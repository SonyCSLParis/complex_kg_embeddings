# -*- coding: utf-8 -*-
"""
Generate classes for ULTRA
"""
import click
from loguru import logger


@click.command()
@click.argument("file_out_p")
def main(file_out_p):
    """ Generate dataset classes for ULTRA """ 

    config_count = 0
    python_content = []
    classes_for_now = []

    for prop in [0, 1]:
        for subevent in [0, 1]:
            for role in [0, 1]:
                for causation in [0, 1]:
                    if causation == 1 or role == 1:
                        syntax_list = ["simple_rdf_reification", "simple_rdf_sp", "simple_rdf_prop"]
                    else:
                        syntax_list = ["simple_rdf_prop"]

                    for syntax in syntax_list:
                        syntax_c = ''.join([x.capitalize() for x in syntax.split("_")])
                        class_name = f"NarrativeDatasetProp{str(prop)}Subevent{str(subevent)}Role{str(role)}Causation{str(causation)}Syntax{syntax_c}"
                        folder_name = f"kg_base_prop_{prop}_subevent_{subevent}_role_{role}_causation_{causation}_syntax_{syntax}"
                        logger.info(f"Configuration {config_count}: Class {class_name}")
                        
                        content = f"""class {class_name}(TransductiveDataset):
    urls = []
    name = "{folder_name}"
    @property
    def raw_file_names(self):
        return [
            "train.txt",
            "valid.txt",
            "test.txt",
        ]"""
                        python_content.append(content)
                        if not role:
                            classes_for_now.append(class_name)
                        config_count += 1
        
        with open(file_out_p, 'w', encoding='utf-8') as f:
            cleaned_content = [content.rstrip() for content in python_content]
            f.write("\n".join(cleaned_content))
        logger.info(f"Classes written to {file_out_p}")
        logger.info(f"Classes for now (0-shot, multi-run): {','.join(classes_for_now)}")
        logger.info(f"Classes for now (fine-tune): {' '.join(classes_for_now)}")


if __name__ == '__main__':
    # python scripts/generate_ultra_classes.py scripts/ultra_classes.txt
    main()
