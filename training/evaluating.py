from training import TestModel, TestFourierModel

def TestAllModel(model, args):
    model.eval()
    
    lofi1 = TestModel(model, 'lofi-track-1-clip-', 720, args.batch_size)
    lofi2 = TestModel(model, 'lofi-track-2-clip-', 444, args.batch_size)
    lec = TestModel(model, 'lecture-clip-', 621, args.batch_size)
    jazz = TestModel(model, 'jazz-clip-', 720, args.batch_size)
    city = TestModel(model, 'city-sounds-clip-', 720, args.batch_size)
    white = TestModel(model, 'white-noise-clip-', 720, args.batch_size)
    
    return ['lofi1','lofi2','lec','jazz','city','white'], [lofi1,lofi2,lec,jazz,city,white]


def TestAllFourierModel(model):
    model.eval()
    
    lofi1 = TestFourierModel(model, 'lofi-track-1-clip-', 2)
    lofi2 = TestFourierModel(model, 'lofi-track-2-clip-', 2)
    lec = TestFourierModel(model, 'lecture-clip-', 2)
    jazz = TestFourierModel(model, 'jazz-clip-', 2)
    city = TestFourierModel(model, 'city-sounds-clip-', 2)
    white = TestFourierModel(model, 'white-noise-clip-', 2)
    
    return ['lofi1','lofi2','lec','jazz','city','white'], [lofi1,lofi2,lec,jazz,city,white]
