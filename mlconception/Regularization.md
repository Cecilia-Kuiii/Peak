# Regularization
L1正则化（Lasso正则化）：对模型的权重绝对值之和0加权，通过使绝大多数参数变为0从而达到特征选择的目的。

L2正则化（Ridge正则化）：对模型的平方和加权，使模型参数更加平滑和稳定，降低过拟合风险。

Dropout正则化：随机地让一些神经元失活，避免特定的神经元对特征的过度依赖，从而增强模型健壮性。

Early stopping：在训练期间监控模型的验证误差，当验证误差开始增加时停止训练。

Max-Norm正则化：限制每个神经元的权重的范数，避免过度依赖一个很小的数量的特征，增加模型泛化性能。

Elastic Net正则化：综合使用L1和L2正则化的方法，融合特征选择和模型稳定性。
