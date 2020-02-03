#include <torch/torch.h>
#include <iostream>
#include "input_transform.h"


InputTransformImpl::InputTransformImpl(int K){
    this->K = K;
    conv_1 = register_module("conv_1", torch::nn::Conv1d(torch::nn::Conv1dOptions(3, 64, 1)
                .padding(0).stride(1).dilation(1).with_bias(true)));

    conv_2 = register_module("conv_2", torch::nn::Conv1d(torch::nn::Conv1dOptions(64, 128, 1)
                .padding(0).stride(1).dilation(1).with_bias(true)));

    conv_3 = register_module("conv_3", torch::nn::Conv1d(torch::nn::Conv1dOptions(128, 1024, 1)
                .padding(0).stride(1).dilation(1).with_bias(true)));

    fc_1 = register_module("fc_1", torch::nn::Linear(1024, 512));
    fc_2 = register_module("fc_2", torch::nn::Linear(512, 256));
    

    bn_1 = register_module("bn_1", torch::nn::BatchNorm(64));
    bn_2= register_module("bn_2", torch::nn::BatchNorm(128));
    bn_3 = register_module("bn_3", torch::nn::BatchNorm(1024));

}

at::Tensor InputTransformImpl::forward(torch::Tensor x){
    x = torch::relu(bn_1->forward(conv_1->forward(x)));
    x = torch::relu(bn_2->forward(conv_2->forward(x)));
    x = torch::relu(bn_3->forward(conv_3->forward(x)));
    
    //1, 1024, 2048 --> 1, 1024
    // std::cout<<"before shape is: "<<x.sizes()<<std::endl;
    x = torch::max_pool1d(x,2048,2,0);
    // std::cout<<"after shape is: "<<x.sizes()<<std::endl;
    x = x.view({1024});
    x = fc_1->forward(x);
    x = fc_2->forward(x);
    auto transform = torch::zeros({256,K*K}, torch::requires_grad()).to(torch::kCUDA);
    transform = torch::matmul(x, transform);
    transform = transform.view({K,K});

    return transform;

}



FeatureTransformImpl::FeatureTransformImpl(int K){
    this->K = K;
    conv_1 = register_module("conv_1", torch::nn::Conv1d(torch::nn::Conv1dOptions(64, 64, 1)
                .padding(0).stride(1).dilation(1).with_bias(true)));

    conv_2 = register_module("conv_2", torch::nn::Conv1d(torch::nn::Conv1dOptions(64, 128, 1)
                .padding(0).stride(1).dilation(1).with_bias(true)));

    conv_3 = register_module("conv_3", torch::nn::Conv1d(torch::nn::Conv1dOptions(128, 1024, 1)
                .padding(0).stride(1).dilation(1).with_bias(true)));

    fc_1 = register_module("fc_1", torch::nn::Linear(1024, 512));
    fc_2 = register_module("fc_2", torch::nn::Linear(512, 256));
    

    bn_1 = register_module("bn_1", torch::nn::BatchNorm(64));
    bn_2= register_module("bn_2", torch::nn::BatchNorm(128));
    bn_3 = register_module("bn_3", torch::nn::BatchNorm(1024));

}

at::Tensor FeatureTransformImpl::forward(torch::Tensor x){
    x = torch::relu(bn_1->forward(conv_1->forward(x)));
    x = torch::relu(bn_2->forward(conv_2->forward(x)));
    x = torch::relu(bn_3->forward(conv_3->forward(x)));
    
    //1, 1024, 2048 --> 1, 1024
    // std::cout<<"before shape is: "<<x.sizes()<<std::endl;
    x = torch::max_pool1d(x,2048,2,0);
    // std::cout<<"after shape is: "<<x.sizes()<<std::endl;
    x = x.view({1024});
    x = fc_1->forward(x);
    x = fc_2->forward(x);
    auto transform = torch::zeros({256, K*K}, torch::requires_grad()).to(torch::kCUDA);
    transform = torch::matmul(x, transform);
    transform = transform.view({K,K});

    return transform;

}