from mcunet.model_zoo import build_model, model_id_list


print(model_id_list)
for model_id in model_id_list:
    model, resolution, desc = build_model(model_id)
    print(desc, resolution)

# print(build_mcunet('mcunet-512kb-in'), )