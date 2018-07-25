import paddle.dataset.uci_housing as uci_housing
import paddle.fluid as fluid

with fluid.scope_guard(fluid.core.Scope()):
    # initialize executor with cpu
    exe = fluid.Executor(place=fluid.CUDAPlace(0))
    # load inference model
    [inference_program, feed_target_names,fetch_targets] =  \
        fluid.io.load_inference_model(uci_housing.fluid_model(), exe)
    # run inference
    result = exe.run(inference_program,
                     feed={feed_target_names[0]: uci_housing.predict_reader()},
                     fetch_list=fetch_targets)
    # print predicted price is $12,273.97
    print 'Predicted price: ${:,.2f}'.format(result[0][0][0] * 1000)
