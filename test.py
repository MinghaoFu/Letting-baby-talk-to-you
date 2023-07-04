def test_donateacry(dataset, model, best_model):
    model.load_state_dict(best_model)