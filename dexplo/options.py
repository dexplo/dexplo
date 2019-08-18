options_dict = {'max_cols': 200,
                'max_rows': 3,
                'max_colwidth': 50,
                'show_tail': False}

_head_method = False

class options_context:

    def __init__(self, **kwargs):
        self.current_options = {}
        self.new_options = kwargs
        for option, value in kwargs.items():
            if option not in options_dict:
                all_options = ', '.join(options_dict)
                raise KeyError(f'The option {option} does not exist. Here are all the possible '
                               f'options choices {all_options}')
            self.current_options[option] = options_dict[option]

    def __enter__(self):
        for option, value in self.new_options.items():
            options_dict[option] = value
        print(options_dict)

    def __exit__(self, *args):
        for option, value in self.current_options.items():
            options_dict[option] = value
