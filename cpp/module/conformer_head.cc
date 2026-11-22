#include "module/conformer_head.h"

namespace MERCURY {
    template<typename T>
    ConformerHead<T>::ConformerHead(int _head_dim) {
        head_dim = _head_dim;
        inv_norm = static_cast<T>(1.0) / static_cast<T>(std::sqrt(double(head_dim)));
    }

    // relative_shift: relative positional encoding shift 연산
    template<typename T>
    std::shared_ptr<Tensor<T>> ConformerHead<T>::relative_shift(const std::shared_ptr<Tensor<T>>& pos_score){
        const int rows = int(pos_score->value.rows());
        const int cols = int(pos_score->value.cols());

        auto tmp = std::make_shared<Tensor<T>>();
        tmp->value.resize(rows, cols + 1);
        tmp->value.setZero();
        tmp->value.rightCols(cols).noalias() = pos_score->value;

        tmp = tmp->reshape(cols + 1, rows);

        auto output = std::make_shared<Tensor<T>>();
        output->value = tmp->value.bottomRows(cols);
        output = output->reshape(rows, cols);
        return output;
    }

    // forwardProjected: content score + relative position score -> softmax -> weighted sum
    template<typename T>
    std::shared_ptr<Tensor<T>> ConformerHead<T>::forwardProjected(std::shared_ptr<Tensor<T>> query,
                                                                  std::shared_ptr<Tensor<T>> key,
                                                                  std::shared_ptr<Tensor<T>> value,
                                                                  std::shared_ptr<Tensor<T>> pos){
        auto qk = std::make_shared<Tensor<T>>();
        qk->value.noalias() = query->value * key->value.transpose();

        auto qp = std::make_shared<Tensor<T>>();
        qp->value.noalias() = query->value * pos->value.transpose();
        qp = relative_shift(qp);

        qk->value.noalias() += qp->value;
        qk->value *= inv_norm;

        std::shared_ptr<Tensor<float>> weighted = Softmax().forward(qk);
        auto out = std::make_shared<Tensor<T>>();
        out->value.noalias() = weighted->value * value->value;
        return out;
    }
    template class ConformerHead<float>;
};
