#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template<>
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template <>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }

template <>
void caffe_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) { cblas_daxpy(N, alpha, X, 1, Y, 1); }

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template void caffe_set<int>(const int N, const int alpha, int* Y);
template void caffe_set<float>(const int N, const float alpha, float* Y);
template void caffe_set<double>(const int N, const double alpha, double* Y);

template <>
void caffe_add_scalar(const int N, const float alpha, float* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <>
void caffe_add_scalar(const int N, const double alpha, double* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
#else
      NO_GPU;
#endif
    } else {
      memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    }
  }
}

template void caffe_copy<int>(const int N, const int* X, int* Y);
template void caffe_copy<unsigned int>(const int N, const unsigned int* X,
    unsigned int* Y);
template void caffe_copy<float>(const int N, const float* X, float* Y);
template void caffe_copy<double>(const int N, const double* X, double* Y);

template <>
void caffe_scal<float>(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

template <>
void caffe_scal<double>(const int N, const double alpha, double *X) {
  cblas_dscal(N, alpha, X, 1);
}

template <>
void caffe_cpu_axpby<float>(const int N, const float alpha, const float* X,
                            const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_cpu_axpby<double>(const int N, const double alpha, const double* X,
                             const double beta, double* Y) {
  cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_add<float>(const int n, const float* a, const float* b,
    float* y) {
  vsAdd(n, a, b, y);
}

template <>
void caffe_add<double>(const int n, const double* a, const double* b,
    double* y) {
  vdAdd(n, a, b, y);
}

template <>
void caffe_sub<float>(const int n, const float* a, const float* b,
    float* y) {
  vsSub(n, a, b, y);
}

template <>
void caffe_sub<double>(const int n, const double* a, const double* b,
    double* y) {
  vdSub(n, a, b, y);
}

template <>
void caffe_mul<float>(const int n, const float* a, const float* b,
    float* y) {
  vsMul(n, a, b, y);
}

template <>
void caffe_mul<double>(const int n, const double* a, const double* b,
    double* y) {
  vdMul(n, a, b, y);
}

template <>
void caffe_div<float>(const int n, const float* a, const float* b,
    float* y) {
  vsDiv(n, a, b, y);
}

template <>
void caffe_div<double>(const int n, const double* a, const double* b,
    double* y) {
  vdDiv(n, a, b, y);
}

template <>
void caffe_powx<float>(const int n, const float* a, const float b,
    float* y) {
  vsPowx(n, a, b, y);
}

template <>
void caffe_powx<double>(const int n, const double* a, const double b,
    double* y) {
  vdPowx(n, a, b, y);
}

template <>
void caffe_sqr<float>(const int n, const float* a, float* y) {
  vsSqr(n, a, y);
}

template <>
void caffe_sqr<double>(const int n, const double* a, double* y) {
  vdSqr(n, a, y);
}

template <>
void caffe_exp<float>(const int n, const float* a, float* y) {
  vsExp(n, a, y);
}

template <>
void caffe_exp<double>(const int n, const double* a, double* y) {
  vdExp(n, a, y);
}

template <>
void caffe_abs<float>(const int n, const float* a, float* y) {
    vsAbs(n, a, y);
}

template <>
void caffe_abs<double>(const int n, const double* a, double* y) {
    vdAbs(n, a, y);
}

unsigned int caffe_rng_rand() {
  return (*caffe_rng())();
}

template <typename Dtype>
Dtype caffe_nextafter(const Dtype b) {
  return boost::math::nextafter<Dtype>(
      b, std::numeric_limits<Dtype>::max());
}

template
float caffe_nextafter(const float b);

template
double caffe_nextafter(const double b);

template <typename Dtype>
void caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
  boost::uniform_real<Dtype> random_distribution(a, caffe_nextafter<Dtype>(b));
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_uniform<float>(const int n, const float a, const float b,
                              float* r);

template
void caffe_rng_uniform<double>(const int n, const double a, const double b,
                               double* r);

template <typename Dtype>
void caffe_rng_gaussian(const int n, const Dtype a,
                        const Dtype sigma, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  boost::normal_distribution<Dtype> random_distribution(a, sigma);
  boost::variate_generator<caffe::rng_t*, boost::normal_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_gaussian<float>(const int n, const float mu,
                               const float sigma, float* r);

template
void caffe_rng_gaussian<double>(const int n, const double mu,
                                const double sigma, double* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, int* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = static_cast<unsigned int>(variate_generator());
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, unsigned int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, unsigned int* r);

template <>
float caffe_cpu_strided_dot<float>(const int n, const float* x, const int incx,
    const float* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

template <>
double caffe_cpu_strided_dot<double>(const int n, const double* x,
    const int incx, const double* y, const int incy) {
  return cblas_ddot(n, x, incx, y, incy);
}

template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y) {
  return caffe_cpu_strided_dot(n, x, 1, y, 1);
}

template
float caffe_cpu_dot<float>(const int n, const float* x, const float* y);

template
double caffe_cpu_dot<double>(const int n, const double* x, const double* y);

template <>
int caffe_cpu_hamming_distance<float>(const int n, const float* x,
                                  const float* y) {
  int dist = 0;
  for (int i = 0; i < n; ++i) {
    dist += __builtin_popcount(static_cast<uint32_t>(x[i]) ^
                               static_cast<uint32_t>(y[i]));
  }
  return dist;
}

template <>
int caffe_cpu_hamming_distance<double>(const int n, const double* x,
                                   const double* y) {
  int dist = 0;
  for (int i = 0; i < n; ++i) {
    dist += __builtin_popcountl(static_cast<uint64_t>(x[i]) ^
                                static_cast<uint64_t>(y[i]));
  }
  return dist;
}

template <>
float caffe_cpu_asum<float>(const int n, const float* x) {
  return cblas_sasum(n, x, 1);
}

template <>
double caffe_cpu_asum<double>(const int n, const double* x) {
  return cblas_dasum(n, x, 1);
}

template <>
void caffe_cpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

template <>
void caffe_cpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  cblas_dcopy(n, x, 1, y, 1);
  cblas_dscal(n, alpha, y, 1);
}
/* changes */

//calculate gamma in kernel function
//K: dimension of data
//S: size of all data
//N: num of output
//W: weights, N*K
//X: X (input of this layer)
template <>
float cal_gamma_cpu<float>(const int K, const int S, const int N, const float* W, const float* X, float* tempX1, float* tempX2){
    srand((unsigned int)time(0));
    float gamma = 0;
    float temp = 0;
    
    //random sample S pair to calculate gamma
    for(int i = 0;i < S;++i){
        memset(tempX1, 0, sizeof(float) * K);
        memset(tempX2, 0, sizeof(float) * K);
        int s1 = rand() % S;
        int s2 = rand() % S;
        s2 = (s1 != s2) ? s2 : (s2 + 1) % S;
        
        const float* x1 = X + s1 * K;
        const float* x2 = X + s2 * K;

        caffe_cpu_gemv<float>(CblasNoTrans, N, K, 1.0, W, x1, 0.0, tempX1);
        caffe_cpu_gemv<float>(CblasNoTrans, N, K, 1.0, W, x2, 0.0, tempX2);
        
        //caffe_cpu_sub<float>(K, tempX1, tempX2, tempX2);
        caffe_cpu_axpby(K, 1.0f, tempX1, -1.0f, tempX2);
        temp = caffe_cpu_dot<float>(K, tempX2, tempX2);
        gamma += temp;
    }
    return S / gamma;
}

//output: 
//  tempX1: W*x1-W*x2
//  tempX2: x1-x2
//  KK: co * (x1-x2)^T * W^T  should be 1*N
template<>
void cal_add_item_cpu<float>(const float co, const int N, const int K, const float* W, const float* x1, float* tempX1, 
    const float* x2, float* tempX2, const float gamma, float* KK){

    memset(tempX1, 0, sizeof(float) * K);
    memset(tempX2, 0, sizeof(float) * K);
    
    caffe_cpu_gemv<float>(CblasNoTrans, N, K, 1.0, W, x1, 0.0, tempX1);
    caffe_cpu_gemv<float>(CblasNoTrans, N, K, 1.0, W, x2, 0.0, tempX2);
    
    float square_sum = 0;

    caffe_cpu_axpby(K, -1.0f, tempX2, 1.0f, tempX1);
    caffe_cpu_axpby(K, 1.0f, x1, 0.0f, tempX2);
    caffe_cpu_axpby(K, -1.0f, x2, 1.0f, tempX2);
    square_sum = caffe_cpu_dot<float>(K, tempX1, tempX1);

    //calculate 2 * \gamma * kernel
    float kernel = 0.0f;
    float tempGamma = gamma / 4.0f;
    for(int i = 0;i < 5;++i){
        float temp = (0.0 - tempGamma) * square_sum;
        temp = exp(temp);
        kernel += 2 * tempGamma* temp;
        tempGamma = tempGamma * 2;
    }
    /*float kernel = (0.0 - gamma) * square_sum;
    kernel = exp(kernel);
    kernel = 2 * gamma * kernel;
*/
    //calculate KK <- co * kernel * X^T * W + 1 * KK
    caffe_cpu_gemm<float>(CblasNoTrans, CblasTrans, 1, N, K, co*kernel, tempX2, W, 0.0, KK);
}

template<>
void cal_add_item_cpu<double>(const double co, const int N, const int K, const double* W, const double* x1, double* tempX1, 
    const double* x2, double* tempX2, const double gamma, double* KK){
    //TODO: complete double version of this function
}

// Gradient with respect to weight for MMD
      //N: number of output neuron
      //K: dimension of the feature
      //M: size of all data
      //S: size of source data in a batch
      //W: weight of this layer
      //X: input of this layer
      //gamma: gamma / learning rate
      //delta_W: gredient of weight

template<>
void caffe_cpu_mmd<float>(const int N, const int K, const int M, const int S, const int labeledTargetSize, 
    const float* W, const float* X, const float gamma, float* delta_W){
    srand((unsigned int)time(0));
    
    //output the value of delta_W before MMD gradient
    float sum = 0;
    for(int i = 0;i < N;++i){
        for(int j = 0;j < K;++j){
            sum += (delta_W[i * N + j] > 0) ? delta_W[i*N+j] : (-1 * delta_W[i*N+j]);
        }
    }
    LOG(INFO) << "delta_W before MMD, sum = " << sum << ", average = " << sum / (N*K);

    float *KK = new float[N];
    float *tempX1 = new float[K];
    float *tempX2 = new float[K];

    float kernel_gamma = cal_gamma_cpu(K, M, N, W, X, tempX1, tempX2);
    int SS = (S>(M-S)) ? S : M-S;
    
    for(int i = 0;i < SS;++i){
        //random
        int s1 = rand() % S;
        int s2 = rand() % S;
        if(s1 == s2){
            s2 = (s2 + 100) % S;
        }
        int t1 = rand() % (M - S - labeledTargetSize);
        int t2 = rand() % (M - S - labeledTargetSize);
        if(t1 == t2){
            t2 = (t2 + 100) % (M - S - labeledTargetSize);
        }
        t1 = t1 + S + labeledTargetSize;
        t2 = t2 + S + labeledTargetSize;
        
        const float *x_s1 = X + s1 * K;
        const float *x_s2 = X + s2 * K;
        const float *x_t1 = X + t1 * K;
        const float *x_t2 = X + t2 * K;
        const float tempS = 1.0;

        //calculate four items of MMD gradient
        memset(KK, 0, sizeof(float) * N);
        cal_add_item_cpu<float>(-1, N, K, W, x_s1, tempX1, x_s2, tempX2, kernel_gamma, KK);
        caffe_cpu_gemm<float>(CblasNoTrans, CblasTrans, N, K, 1, tempS * gamma, KK,tempX2, 1.0, delta_W);

        memset(KK, 0, sizeof(float) * N);
        cal_add_item_cpu<float>(1, N, K, W, x_s1, tempX1, x_t2, tempX2, kernel_gamma, KK);
        caffe_cpu_gemm<float>(CblasNoTrans, CblasTrans, N, K, 1, tempS * gamma, KK,tempX2, 1.0, delta_W);

        memset(KK, 0, sizeof(float) * N);
        cal_add_item_cpu<float>(1, N, K, W, x_s2, tempX1, x_t1, tempX2, kernel_gamma, KK);
        caffe_cpu_gemm<float>(CblasNoTrans, CblasTrans, N, K, 1, tempS * gamma, KK,tempX2, 1.0, delta_W);

        memset(KK, 0, sizeof(float) * N);
        cal_add_item_cpu<float>(-1, N, K, W, x_t1, tempX1, x_t2, tempX2, kernel_gamma, KK);
        caffe_cpu_gemm<float>(CblasNoTrans, CblasTrans, N, K, 1, tempS * gamma, KK,tempX2, 1.0, delta_W);
    }
    //output the value of delta_W after MMD gradient
    sum = 0;
    for(int i = 0;i < N;++i){
        for(int j = 0;j < K;++j){
            sum += (delta_W[i * N + j] > 0) ? delta_W[i*N+j] : (-1 * delta_W[i*N+j]);
        }
    }
    LOG(INFO) << "delta_W after MMD, sum = " << sum << ", average = " << sum / (N*K);

    delete [] KK;
    delete [] tempX1;
    delete [] tempX2;
}

template<>
void caffe_cpu_mmd<double>(const int N, const int K, const int M, const int S, const int labeledTargetSize,
    const double* W, const double* X, const double gamma, double* delta_W){
    //TODO: complete the double version of this function
}

int compare_float_cpu(const void* a, const void* b){
    float arg1 = *reinterpret_cast<const float*>(a);
    float arg2 = *reinterpret_cast<const float*>(b);
    if(arg1 < arg2) return -1;
    if(arg1 > arg2) return 1;
    return 0;
}

float find_topK_cpu(float* arr, const int k, const int len){
    std::qsort(arr, len, sizeof(float), compare_float_cpu);
    return arr[k];
}

//Gradient of GR
  //N: output num
  //K: input dimension
  //M: minibatch size
  //W: weight
  //X: input
  //topK: top-k
  //lambda: lambda
  //delta_W: gradient
template<>
void caffe_cpu_GR<float>(const int N, const int K, const int M, const float* W, 
    const float* X, const int topK, const float lambda, float* delta_W){
    float *S = new float[M * M];
    float *distance = new float[M * M];
    float *D = new float[M];
    float *topK_thr = new float[M];
    float *temp = new float[M];
    float *WX = new float[N * M];
    float *WXL = new float[N * M];

    memset(distance, 0, sizeof(float) * M * M);
    //calculate distance of X_i, X_j
    for(int i = 0;i < M;++i){
        for(int j = i + 1;j < M;++j){
            const float* x1 = X + K * i;
            const float* x2 = X + K * j;
            float square_sum = 0;
            for(int k = 0;k < K;++k){
                square_sum += (x1[k] - x2[k]) * (x1[k] - x2[k]);
            }
            distance[i * M + j] = square_sum;
            distance[j * M + i] = square_sum;
        }
    }
    
    //calculate threshold of topK
    for(int i = 0;i < M;++i){
        memcpy(temp, distance + i*M, sizeof(float) * M);
        topK_thr[i] = find_topK_cpu(temp, topK, M);
    }

    //init S
    for(int i = 0;i < M;++i){
        for(int j = 0;j < M;++j){
            S[i * M + j] = (distance[i * M + j] <= topK_thr[i]) ? 1 : 0;
        }
    }

    //S <- (S + S^T) / 2
    for(int i = 0;i < M;++i){
        for(int j = i + 1;j < M;++j){
            S[i * M + j] = (S[i * M + j] + S[j * M + i]) / 2;
            S[j * M + i] = S[i * M + j];
        }
    }

    //calculate D
    for(int i = 0;i < M;++i){
        float sum = 0;
        for(int j = 0;j < M;++j){
            sum += S[i * M + j];
        }
        D[i] = sum;
    }
    
    //calculate L(store in S)
    for(int i = 0;i < M;++i){
        for(int j = 0;j < M;++j){
            if(i == j) S[i * M + j] = D[i] - S[i * M + j];
            else S[i * M + j] = 0 - S[i * M + j];
        }
    }
    
    memset(WX, 0, sizeof(float) * N * M);
    memset(WXL, 0, sizeof(float) * N * M);

    //W*X
    caffe_cpu_gemm<float>(CblasNoTrans, CblasTrans, N, M, K, 1.0, W, X, 0.0, WX);
    //W*X*L
    caffe_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, N, M, M, 1.0, WX, S, 0.0, WXL);
    //delta_W <- lambda * W*X*L*X^T + delta_W
    caffe_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, N, K, M, lambda, WXL, X, 1.0, delta_W);

    //output the value of delta_W after GR gradient
    float sum = 0;
    for(int i = 0;i < N;++i){
        for(int j = 0;j < K;++j){
            sum += delta_W[i * N + j];
        }
    }
    LOG(INFO) << "GR delta_W sum " << sum << " average " << sum / (N*K);

    delete [] S;
    delete [] distance;
    delete [] D;
    delete [] topK_thr;
    delete [] temp;
    delete [] WX;
    delete [] WXL;
}

template<>
void caffe_cpu_GR<double>(const int N, const int K, const int M, const double* W, 
    const double* X, const int topK, const double lambda, double* delta_W){
    //TODO: complete the double version of this function
}
/* end of change */
}  // namespace caffe
