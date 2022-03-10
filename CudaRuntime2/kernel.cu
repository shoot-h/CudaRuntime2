//CUDAプログラミングによる差分法を用いたラプラス方程式の解くプログラム
//並列化により，計算時間がどれだけ短縮するかを計測する
//初期条件はすでに与えられているものとする
//参考：https://www2.akita-nct.ac.jp/saka/Lecturenote/lecture/5e/text/26,27.pdf

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

# define hSize 2048

//差分法によるラプラス方程式計算
__global__ void Laplace(float* u,int* flag)
{
    int i = threadIdx.x;
    for (int j = 0; j < 10; j++) {
        if (flag[i] == 0) {
            u[i] = 0.25 * (u[i + 1] + u[i - 1] + u[i + hSize] + u[i - hSize]);
        }
    }
} 

//初期化関数
void Initialize(float* u,int* flag) 
{
    //境界条件設定
    for (int i = 0; i < hSize; i++) {
        flag[i * hSize] = 1;
        flag[(i * hSize) + hSize - 1] = 1;
        flag[i] = 1;
        flag[i + (hSize - 1) * hSize] = 1;
        u[i * hSize] = 0;
        u[(i * hSize) + hSize - 1] = 0;
        u[i] = 0;
        u[i + (hSize - 1) * hSize] = 0;
    }

    //初期ポテンシャル決定
    for (int i = 1; i < hSize - 1; i++) {
        for (int j = 1; j < hSize - 1; j++) {
            if (((i - (0.25 * hSize)) * (i - (0.25 * hSize))) + ((j - (0.75 * hSize)) * (j - (0.75 * hSize))) <= 0.125 * hSize * 0.125 * hSize) {
                flag[i + (j * hSize)] = 1;
                u[i + (j * hSize)] = 100;
            }
            else if (((i - (0.875 * hSize)) * (i - (0.875 * hSize))) + ((j - (0.125 * hSize)) * (j - (0.125 * hSize))) <= 0.05 * hSize * 0.05 * hSize) {
                flag[i + (j * hSize)] = 1;
                u[i + (j * hSize)] = 20;
            }
            else {
                flag[i + (j * hSize)] = 0;
                u[i + (j * hSize)] = 0;
            }

        }
    }
}

//結果出力関数
void OutputResult(float calcTime,float* u) 
{
    //計算時間出力
    printf("%f\n", calcTime);

    /*for (int i = 0; i < h; i++) {
        for (int j = h - 1; j >= 0; j--) {
            printf("%f\t", u[i + (j * h)]);
        }
        printf("\n");
    } //計算できているか確認用*/
}

int main(int argc,char* argv[])
{
    float u[hSize*hSize]; //ポテンシャル保存用配列
    int flag[hSize*hSize]; //固定ポテンシャルかを判別する配列(固定：1，可変：0)

    //計算時間計測用
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float calcTime = 0;

    //GPU上にメモリ確保
    float* d_u;
    int* d_flag;
    size_t floatsize = hSize * hSize * sizeof(float);
    size_t intsize = hSize * hSize * sizeof(int);
    cudaMalloc((void**)&d_u, floatsize);
    cudaMalloc((void**)&d_flag, floatsize);

    Initialize(u,flag);//初期化

    //GPUメモリにデータコピー
    cudaMemcpy(d_flag, flag, intsize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u, floatsize, cudaMemcpyHostToDevice);

    //グリッドとブロック指定
    const int threadsPerGrid = 1;
    const int threadsPerBlock = hSize*hSize;

    //ラプラス方程式計算計算
    cudaEventRecord(start);
    Laplace << <threadsPerGrid, threadsPerBlock >> > (d_u,d_flag);
    cudaMemcpy(u, d_u, floatsize, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&calcTime, start, stop);//計算時間算出

    OutputResult(calcTime,u);//結果出力

    //メモリ開放
    cudaEventDestroy(start);cudaEventDestroy(stop);cudaFree(d_u);cudaFree(d_flag);

    return 0;
}
