#include "tensor/tensor.h"
#include <iomanip>

namespace MERCURY {
    // ~Tensor() = default is declared in tensor.h (Rule of Five)

    template<typename T>
    void Tensor<T>::loadValue(std::string path, std::vector<int> _shape) {  // NOLINT: value params intentional for load-time API
        // Check that _shape has at least two dimensions
        // LoggerManager::Log("INFO", std::string("START LOAD VALUE :" )+ path.c_str());
        // 여기서 실패하면 반드시 던진다.
        //
        // 예전에는 로그만 남기고 그냥 돌아갔다. 그러면 값이 비어 있는 텐서가 그대로
        // 모델에 실려 나가고, 뒤따르는 연산이 그것을 만지다 널 역참조로 프로세스째
        // 죽는다 — 무엇이 잘못됐는지 로그에 다 적어 놓고도 그렇게 된다.
        // (Galaxy S9: 텐서 하나가 786432 대신 472064 바이트로 잘려 있어 SIGSEGV.)
        //
        // 던지면 AsrAPI 생성자까지 전파되고, 브리지가 그것을 받아 "엔진을 만들지 못했다"
        // 로 돌려준다(iOS MercuryAsr.mm 은 nil, 안드로이드는 초기화 실패 메시지).
        if (_shape.size() < 2) {
            LoggerManager::Log("ERROR", "Invalid shape size: " + path);
            throw std::runtime_error("tensor shape must have at least 2 dims: " + path);
        }
        if (_shape[0] <= 0 || _shape[1] <= 0 ||
            _shape[0] > std::numeric_limits<int>::max() / _shape[1])
            throw std::runtime_error("invalid tensor dimensions: " + path);
        int size = _shape[0] * _shape[1];
        std::ifstream ifs(path, std::ifstream::binary | std::ifstream::ate);
        if (!ifs) {
            LoggerManager::Log("ERROR", std::string("Failed to open file: ") +  path);
            throw std::runtime_error("cannot open tensor file: " + path);
        }
        std::ifstream::pos_type fileSize = ifs.tellg();
        auto expected = static_cast<std::ifstream::pos_type>(size * sizeof(T));
        if (fileSize < expected) {
            // 크기를 함께 적는다 — 어느 파일이 얼마나 모자란지 알아야 고칠 수 있다.
            LoggerManager::Log("ERROR", "File size is smaller than expected data size: " + path
                                       + " (" + std::to_string(static_cast<long long>(fileSize))
                                       + " < " + std::to_string(static_cast<long long>(expected)) + ")");
            throw std::runtime_error("truncated tensor file: " + path);
        }
        ifs.seekg(0, std::ifstream::beg);
        // Read data
        std::vector<T> _value;
        try {
            _value.resize(size);
        } catch (const std::bad_alloc&) {
            LoggerManager::Log("ERROR", "Memory allocation failed: " + path);
            throw;
        }

        ifs.read(reinterpret_cast<char*>(_value.data()), size * sizeof(T));
        if (ifs.fail()) {
            LoggerManager::Log("ERROR", std::string("Failed to read file: ") +  path);
            throw std::runtime_error("cannot read tensor file: " + path);
        }

        // Set value
        setValue(_value, _shape);
    }

    // template<typename T>
    // bool Tensor<T>::empty() {
    //     if(value.size() == 0){
    //         return true;
    //     }
    //     return false;
    // }

    template<typename T>
    void Tensor<T>::setValue(std::vector<T> _value, std::vector<int> _shape) {
        // Direct assignment from Map handles resize internally — no need for explicit resize
        value = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(_value.data(), _shape[0], _shape[1]);
    }

    template<typename T>
    void Tensor<T>::printValue() {
        using std::cout; using std::endl;
        int rows = value.rows();
        int cols = value.cols();

        auto printRow=[&](int r){
            cout << "[";
            int firstCols = std::min(3, cols);
            int lastCols  = (cols > 6) ? 3 : std::max(0, cols - firstCols);

            // first part
            for(int c=0;c<firstCols;++c){
                cout << std::setw(8) << std::setprecision(4) << std::fixed << value(r,c);
                if(c != cols-1) cout << ", ";
            }
            // ellipsis & tail part
            if(cols > 6){
                cout << ", ... , ";
                for(int c=cols-lastCols; c<cols; ++c){
                    cout << std::setw(8) << std::setprecision(4) << std::fixed << value(r,c);
                    if(c != cols-1) cout << ", ";
                }
            }
            cout << "]" << endl;
        };

        cout << "print value" << endl;
        int topRows = std::min(3, rows);
        int bottomRows = (rows > 6) ? 3 : std::max(0, rows - topRows);

        for(int r=0;r<topRows;++r) printRow(r);
        if(rows > 6){
            cout << "..." << endl;
            for(int r=rows-bottomRows; r<rows; ++r) printRow(r);
        } else {
            for(int r=topRows; r<rows; ++r) printRow(r);
        }

        cout << "print shape" << endl;
        cout << "[" << rows << " , " << cols << "]" << endl;
    }

    template<typename T>
    void Tensor<T>::printShape() {
        std::cout << "print shape" << std::endl;
        std::cout << "[" << value.rows() << " , " << value.cols() << "]" << std::endl;
    }

    template<typename T>
    int Tensor<T>::shape(int idx) {
        return idx == 0 ? value.rows() : value.cols();
    }

    template<typename T>
    void Tensor<T>::clone(std::shared_ptr<Tensor<T>> other) {
        // Direct assignment already performs a deep copy for Eigen matrices
        value = other->value;
    }

    template<typename T>
    void Tensor<T>::setRows(int i, std::shared_ptr<Tensor<T>> other, int j) {
        if (value.size()  == 0){
            LoggerManager::Log("ERROR", "Not initialized Matrix");
            assert(false);    
        }
        value.row(i) = other->value.row(j);
    }

    template<typename T>
    void Tensor<T>::setCols(int i, std::shared_ptr<Tensor<T>> other, int j) {
        if (value.size()  == 0){
            LoggerManager::Log("ERROR", "Not initialized Matrix");
            assert(false);    
        }
        value.col(i) = other->value.col(j);
    }
    
    template<typename T>
    std::shared_ptr<Tensor<T>> Tensor<T>::transpose() {
        auto out = std::make_shared<Tensor<T>>();
        // .transpose() returns an expression; assignment evaluates and copies
        out->value = value.transpose();
        return out;
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> Tensor<T>::matmul(std::shared_ptr<Tensor<T>> other) {
        std::shared_ptr<Tensor<T>> out = std::make_shared<Tensor<T>>();
        out->value.noalias() = value * other->value;
        return out;
    }

    /*
    template<typename T>
    std::shared_ptr<Tensor<T>> Tensor<T>::sparse_matmul(std::shared_ptr<Tensor<T>> other) {
        std::shared_ptr<Tensor<T>> out = std::make_shared<Tensor<T>>();
        Eigen::SparseMatrix<float> sparse_self(value.rows(), valuse.col()); 
        Eigen::SparseMatrix<float> sparse_other(other->value.rows(), valuse.col()); 
        out->value = value * other->value;
        return out;
    }
    */

    template<typename T>
    void Tensor<T>::appendRow(std::shared_ptr<Tensor<T>> other) {
        if(value.size() == 0){
            value.resize(1, other->value.cols());
            value.row(0) = other->value;
        }
        else{
            assert(value.cols() == other->value.cols());
            const int old_rows = value.rows();
            value.conservativeResize(old_rows + 1, value.cols());
            value.row(old_rows) = other->value;
        }
    }

    template<typename T>
    void Tensor<T>::appendChunkRow(std::shared_ptr<Tensor<T>> other) {
        const int other_rows = other->shape(0);
        const int other_cols = other->shape(1);
        if(value.size() == 0){
            // Direct assignment avoids separate resize + copy
            value = other->value;
        }
        else{
            assert(value.cols() == other->value.cols());
            const int old_rows = value.rows();
            value.conservativeResize(old_rows + other_rows, value.cols());
            value.block(old_rows, 0, other_rows, other_cols).noalias() = other->value;
        }
    }

    // [2026-04-07] appendChunkRowWithCapacity: capacity 기반 기하급수적 확장 (2배)
    // 기존 appendChunkRow: conservativeResize가 매번 전체 데이터 재할당+복사 → O(N^2)
    // 신규: capacity 부족 시만 2배 확장 → amortized O(1) append, 총 O(N)
    template<typename T>
    void Tensor<T>::appendChunkRowWithCapacity(std::shared_ptr<Tensor<T>> other, int& logical_rows) {
        const int other_rows = other->shape(0);
        const int other_cols = other->shape(1);

        if (logical_rows == 0 && value.size() == 0) {
            int initial_capacity = std::max(other_rows * 2, 64);
            value.resize(initial_capacity, other_cols);
            value.block(0, 0, other_rows, other_cols).noalias() = other->value;
            logical_rows = other_rows;
        } else {
            assert(value.cols() == other_cols);
            int needed = logical_rows + other_rows;
            int current_capacity = value.rows();

            if (needed > current_capacity) {
                int new_capacity = std::max(needed, current_capacity * 2);
                value.conservativeResize(new_capacity, other_cols);
            }
            value.block(logical_rows, 0, other_rows, other_cols).noalias() = other->value;
            logical_rows += other_rows;
        }
    }

    // [2026-04-07] trimToLogicalRows: capacity 기반 append 후 실제 유효 행만 남기고 축소
    template<typename T>
    void Tensor<T>::trimToLogicalRows(int logical_rows) {
        if (logical_rows < value.rows()) {
            value.conservativeResize(logical_rows, value.cols());
        }
    }

    template<typename T>
    void Tensor<T>::halfDimValue(std::shared_ptr<Tensor<T>> other1, std::shared_ptr<Tensor<T>> other2) {
        if(value.size() == 0 || value.cols() % 2 != 0){
            LoggerManager::Log("ERROR", "invalid size of shape");
            assert(false);
        }

        int half_dim = value.cols()/2;
        other1->value = value.block(0, 0, value.rows(), half_dim);
        other2->value = value.block(0, half_dim, value.rows(), half_dim);
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> Tensor<T>::flatten() {
        std::shared_ptr<Tensor<T>> out = std::make_shared<Tensor<T>>();
        out->value = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(value.data(), 1, shape(0)*shape(1));
        return out;
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> Tensor<T>::reshape(int h, int w) {
        std::shared_ptr<Tensor<T>> out = std::make_shared<Tensor<T>>();
        out->value = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(value.data(), h, w);
        return out;
    }

    template class Tensor<float>;
    template class Tensor<int>;
};
