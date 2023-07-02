#include <chrono>
#include <iostream>
#include <CL/sycl.hpp>

#define random_float() (rand() / double(RAND_MAX))

using namespace std;
using namespace sycl;

float matrix[10000][10000];
float matrix1[10000][10000];
void Initialize(int N)
{
    for (int i = 0; i < N; i++)
    {
        //首先将全部元素置为0，对角线元素置为1
        for (int j = 0; j < N; j++)
        {
            matrix[i][j] = 0;
            matrix1[i][j] = 0;
        }
        matrix[i][i] = matrix1[i][i] = 1.0;
        //将上三角的位置初始化为随机数
        for (int j = i + 1; j < N; j++)
        {
            matrix[i][j] = matrix1[i][j] = rand();
        }
    }
    for (int k = 0; k < N; k++)
    {
        for (int i = k + 1; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                matrix[i][j] += matrix[k][j];
                matrix1[i][j] += matrix1[k][j];
            }
        }
    }
}

void cpu_kernel(int N)
{
    double duration = 0.0;
    std::chrono::high_resolution_clock::time_point s, e;
    //开始计时
    s = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < N; k++) {
        for (int j = k + 1; j < N; j++)
            matrix[k][j] /= matrix[k][k];
        matrix[k][k] = 1.0;
        for (int i = k + 1; i < N; i++) {
            for (int j = k + 1; j < N; j++)
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            matrix[i][k] = 0;
        }
    }

    //结束计时
    e = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<float, std::milli>(e - s).count();
    std::cout << "cpu time= " << duration << std::endl;
}


void gpu_kernel(int N)
{
    double duration = 0.0;
    std::chrono::high_resolution_clock::time_point s, e;
    //开始计时
    s = std::chrono::high_resolution_clock::now();

    queue my_gpu_queue(cl::sycl::gpu_selector{});//创建队列
    // 创建缓冲区来存储输入数据
    sycl::buffer<float, 2> buf(sycl::range<2>(N, N));

    // 将全局变量的二维数组加载到缓冲区
    q.submit([&](sycl::handler& h) {
        // 创建访问器并初始化它，加载到加速器，赋予读写权限
        sycl::accessor<float, 2, sycl::access::mode::read_write> m{ buf, h，read_write };
        }
        //进行并行计算
    int n = buf.get_range()[0];
    for (int k = 0; k < n; k++) {

        q.submit([&](handler& h) {
            accessor m{ buf, h, read_write };
            h.parallel_for(range(n - k), [=](auto idx) {
                int j = k + idx;
                m[k][j] = m[k][j] / m[k][k];
                });
            });

        q.submit([&](handler& h) {
            accessor m{ buf, h, read_write };
            h.parallel_for(range(n - (k + 1), n - (k + 1)), [=](auto idx) {
                int i = k + 1 + idx.get_id(0);
                int j = k + 1 + idx.get_id(1);
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
                });
            });

        q.submit([&](handler& h) {
            accessor m{ buf, h, read_write };
            h.parallel_for(range(n - (k + 1)), [=](auto idx) {
                int i = k + 1 + idx;
                m[i][k] = 0;
                });
            });
    }
    q.wait();

    //结束计时
    e = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<float, std::milli>(e - s).count();
    std::cout << "gpu time= " << duration << std::endl;

    // 从缓冲区中获取访问器并访问数据
    auto accInput = buf.get_access<sycl::access::mode::read>();

    // 将数据从缓冲区复制到内存
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            matrix1[i][j] = accInput[i][j];
        }
    }
}

int main() {
    Initialize(1000);
    cpu_kernel(1000);
    gpu_kernel(1000);
}
