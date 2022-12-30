from torch import nn


def safe_load_dict(model, new_model_state, should_resume_all_params=False):
    old_model_state = model.state_dict()
    c = 0
    if should_resume_all_params:
        for old_name, old_param in old_model_state.items():
            assert old_name in list(new_model_state.keys()) or old_name[old_name.find('.') + 1:] in list(new_model_state.keys()), "{} parameter is not present in resumed checkpoint".format(
                old_name)
    else:
        for old_name, old_param in old_model_state.items():
            if not (old_name in list(new_model_state.keys()) or old_name[old_name.find('.') + 1:] in list(new_model_state.keys())):
                print(f'Warning: {old_name} model parameters failed to load')

    for name, param in new_model_state.items():
        n = name.split('.')
        beg = n[0]
        end = n[1:]
        if beg == 'module':
            name = '.'.join(end)
        if name not in old_model_state:
            if f'model.{name}' in old_model_state:
                name = f'model.{name}'
            elif name[name.find('.') + 1:] in old_model_state:
                name = name[name.find('.') + 1:]
            else:
                # print('%s not found in old model.' % name)
                continue
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        c += 1
        if old_model_state[name].shape != param.shape:
            print('Shape mismatch...ignoring %s' % name)
            continue
        else:
            old_model_state[name].copy_(param)
    if c == 0:
        raise AssertionError('No previous ckpt names matched and the ckpt was not loaded properly.')