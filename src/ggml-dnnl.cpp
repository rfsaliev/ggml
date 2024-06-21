#include "ggml.h"
#include "ggml-dnnl.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <functional>
#include <tuple>
#include <unordered_map>
#include <utility>

#include <dnnl.hpp>

#define USE_DNNL_BUFFER 0

static const char* ggml_backend_dnnl_name_str = "DNNL";

struct ggml_dnnl_context {
    dnnl::engine engine;
    dnnl::stream stream;

    ggml_dnnl_context(size_t dev_idx)
        : engine{dnnl::engine::kind::cpu, dev_idx}
        , stream{engine}
    {}
};

#if USE_DNNL_BUFFER
static const size_t    DNNL_BUFFER_ALIGNMENT = 0x40;
static const uintptr_t DNNL_BUFFER_BASE = 0x1000;

struct ggml_backend_dnnl_buffer_context
{
    dnnl::memory mem;
    std::vector<dnnl::memory> sub_mems;
};

static void* get_memory_handle(const struct ggml_tensor * t) {
    auto buf_mem = dnnl::memory{(dnnl_memory_t)t->extra, true};
    auto buf_md = buf_mem.get_desc();
    auto buf_handle = buf_mem.get_data_handle();
    auto buf_offset = buf_md.get_submemory_offset();
    GGML_ASSERT((size_t)buf_offset == ((uintptr_t)t->data - DNNL_BUFFER_BASE));
//    auto buf_ctx = (ggml_backend_dnnl_buffer_context*)t->buffer->context;
    //auto parent_buf_handle = buf_ctx->mem.get_data_handle();

    // FIXME: buf_handle + offset works for CPU only
    return (char*)buf_handle + buf_offset;
}
#else
static void* get_memory_handle(const struct ggml_tensor * t) {
    return t->data;
}
#endif


namespace {
template <class T>
struct dnnl_mem_ptr {
    dnnl_mem_ptr(const dnnl::memory& mem)
     : mem_{mem}
     , ptr_{mem_.map_data<T>()}
     , offset_{mem_.get_desc().get_submemory_offset()} {
    }
    ~dnnl_mem_ptr() {
        mem_.unmap_data(ptr_);
    }

    const T* get() const {return (const T*)((const char*)ptr_ + offset_);}
    T* get() {return (T*)((char*)ptr_ + offset_);}
    operator const T*() const {return get();}
    operator T*() {return get();}
private:
    dnnl::memory mem_;
    T* ptr_;
    dnnl::memory::dim offset_;
};

template <class T = void>
dnnl_mem_ptr<T> map_memory(const dnnl::memory& mem) {
     return dnnl_mem_ptr<T>{mem};
}
} // namespace

static dnnl::memory::data_type ggml_type_to_dnnl_dtype(enum ggml_type type) {
    using dt = dnnl::memory::data_type;
    switch (type) {
        case GGML_TYPE_F32:
            return dt::f32;
        case GGML_TYPE_F16:
            return dt::f16;
        case GGML_TYPE_I8:
            return dt::s8;
        case GGML_TYPE_I32:
            return dt::s32;
        case GGML_TYPE_F64:
            return dt::f64;
        // case GGML_TYPE_BF16:
        //     return dt::bf16;
        default:
            return dt::undef;
    }
}

static bool ggml_dnnl_type_supported(enum ggml_type type) {
    return ggml_type_to_dnnl_dtype(type) != dnnl::memory::data_type::undef;
}

static bool ggml_dnnl_tensor_supported(const struct ggml_tensor * t) {
    auto type = t->type;

    if (!ggml_dnnl_type_supported(type)) {
        return false;
    }
    return true;
}

static bool are_same_descs(const struct ggml_tensor * t0, const struct ggml_tensor * t1) {
    return t0->type == t1->type
           && ggml_are_same_shape(t0, t1)
           && ggml_are_same_stride(t0, t1);
}

static bool unary_tensor_supported(ggml_backend_t, const ggml_tensor* op) {
    return are_same_descs(op, op->src[0])
           && ggml_dnnl_tensor_supported(op)
           && ggml_dnnl_tensor_supported(op->src[0]);
}

static bool binary_tensor_supported(ggml_backend_t, const ggml_tensor* op) {
    return ggml_dnnl_tensor_supported(op)
           && ggml_dnnl_tensor_supported(op->src[0])
           && ggml_dnnl_tensor_supported(op->src[1]);
}

static bool ggml_compute_forward_mul_mat_use_dnnl(ggml_backend_t, const struct ggml_tensor * dst) {
#if 0
    return true;
    GGML_UNUSED(dst);
#else
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    bool ok = dst->op != GGML_OP_MUL_MAT_ID
            && ggml_dnnl_tensor_supported(src0)
            && ggml_dnnl_tensor_supported(src1)
            && ggml_dnnl_tensor_supported(dst)
            && ggml_is_contiguous(src0)
            && ggml_is_contiguous(src1);
    if (!ok) {
        return false;
    }

    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    // TODO: find the optimal values for these
    if (ne0 >= 32 && ne1 >= 32 && ne10 >= 32) {

        /*printf("BLAS: %d %d %d %d %d\n", ne0, ne1, ne10, ne00, ne01);*/
        return true;
    }

    // OneDNN support only 1->N broadcasts
    for (size_t i = 2; i < GGML_MAX_DIMS; ++i) {
        if (src0->ne[i] != src1->ne[i] && src0->ne[i] != 1 && src1->ne[i] != 1) {
            return false;
        }
    }

    return ok;
#endif
}

dnnl::memory::desc ggml_tensor_to_dnnl_md(const struct ggml_tensor * t, bool transpose = false,
                                          dnnl::memory::data_type dtype = dnnl::memory::data_type::undef,
                                          size_t ndims = GGML_MAX_DIMS) {
    GGML_ASSERT(ggml_dnnl_tensor_supported(t));
    GGML_ASSERT(ndims > 0);
    using dims_t = dnnl::memory::dims;

    const auto tensor_type = t->type;
    auto dt = dtype != dnnl::memory::data_type::undef
              ? dtype : ggml_type_to_dnnl_dtype(tensor_type);
    auto type_size = ggml_type_size(tensor_type);

    dims_t adims(ndims);
    dims_t strides(ndims);

    for (size_t i = 0; i < ndims; i++ ) {
        adims[ndims - 1 - i] = t->ne[i];
        strides[ndims - 1 - i] = t->nb[i] / type_size;
    }

    if (transpose) {
        std::swap(adims[ndims-1], adims[ndims-2]);
        std::swap(strides[ndims-1], strides[ndims-2]);
    }

    for (size_t i = ndims; i < GGML_MAX_DIMS; i++) {
        GGML_ASSERT(t->nb[i] == t->nb[i-1] * t->ne[i-1]);
        adims[0] *= t->ne[i];
    }

    return dnnl::memory::desc{adims, dt, strides};
}

dnnl::memory ggml_tensor_to_dnnl_mem(ggml_backend_t backend, const struct ggml_tensor * t, bool transpose = false,
                                           dnnl::memory::data_type convert_to = dnnl::memory::data_type::undef,
                                           size_t ndims = GGML_MAX_DIMS) {
    auto * ctx = static_cast<ggml_dnnl_context *>(backend->context);

    auto t_md = ggml_tensor_to_dnnl_md(t, transpose, dnnl::memory::data_type::undef, ndims);
    auto t_mem = dnnl::memory{t_md, ctx->engine, get_memory_handle(t)};

    auto dst_md = ggml_tensor_to_dnnl_md(t, transpose, convert_to, ndims);
    if (t_md.get_data_type() == dst_md.get_data_type()) {
        return t_mem;
    }

    // else convert to requested type
    auto dst_mem = dnnl::memory{dst_md, ctx->engine};
    auto reorder = dnnl::reorder{t_mem, dst_mem};
    reorder.execute(ctx->stream, t_mem, dst_mem);
    ctx->stream.wait();
    return dst_mem;
}

static bool adjust_for_dnnl_broadcast(dnnl::memory::dims& src0, dnnl::memory::dims& src1, dnnl::memory::dims& dst, size_t n_dims_to_keep = 0) {
    GGML_ASSERT(src0.size() == src1.size() && src0.size() == dst.size());

    bool changed = false;
    for (size_t i = 0; i < dst.size() - n_dims_to_keep; i++) {
        if (src0[i] != src1[i] && src0[i] != 1 && src1[i] != 1){
            // split dims
            auto gcd = std::min(src0[i], src1[i]);
            GGML_ASSERT(std::max(src0[i], src1[i]) % gcd == 0);
            src0[i] /= gcd;
            src1[i] /= gcd;
             dst[i] /= gcd;
            ++i;
            src0.insert(src0.begin() + i, gcd);
            src1.insert(src1.begin() + i, gcd);
             dst.insert( dst.begin() + i, gcd);
            changed = true;
        }
    }
    return changed;
}

static ggml_status ggml_backend_dnnl_inner_product(ggml_backend_t backend, struct ggml_tensor * dst) {
    auto * ctx = static_cast<ggml_dnnl_context *>(backend->context);

    // NOTE: src0 - weights, src1 - input
    const struct ggml_tensor * src = dst->src[1];
    const struct ggml_tensor * weights = dst->src[0];

    GGML_TENSOR_LOCALS(int64_t, ne_s, src, ne)
    GGML_TENSOR_LOCALS(int64_t, ne_w, weights, ne)
    GGML_TENSOR_LOCALS(int64_t, ne_d, dst, ne)

    GGML_ASSERT(ne_d0 == ne_w1);
    GGML_ASSERT(ne_d1 == ne_s1);
    GGML_ASSERT(ne_d2 == ne_s2);
    GGML_ASSERT(ne_d3 == ne_s3);
    GGML_ASSERT(ne_w2 == 1 && ne_w3 == 1);

    auto dst_mem = ggml_tensor_to_dnnl_mem(backend, dst, false, dnnl::memory::data_type::undef, 2);
    auto dst_md     = dst_mem.get_desc();
    auto src_mem = ggml_tensor_to_dnnl_mem(backend, src, false, dst_md.get_data_type(), 2);
    auto src_md     = src_mem.get_desc();
    auto weights_mem = ggml_tensor_to_dnnl_mem(backend, weights, false, dst_md.get_data_type(), 2);
    auto weights_md = weights_mem.get_desc();

    auto pd = dnnl::inner_product_forward::primitive_desc{ctx->engine, dnnl::prop_kind::forward_inference, src_md, weights_md, dst_md};
    auto prim = dnnl::inner_product_forward{pd};

    std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC,     src_mem},
        {DNNL_ARG_WEIGHTS, weights_mem},
        {DNNL_ARG_DST,     dst_mem},
    };

    prim.execute(ctx->stream, args);
    ctx->stream.wait();
    return GGML_STATUS_SUCCESS;
}


static ggml_status ggml_backend_dnnl_mul_mat(ggml_backend_t backend, struct ggml_tensor * dst) {
    auto * ctx = static_cast<ggml_dnnl_context *>(backend->context);

    // NOTE: src0 - weights, src1 - input
    const struct ggml_tensor * src = dst->src[1];
    const struct ggml_tensor * weights = dst->src[0];
    
    if (weights->ne[2] == 1 && weights->ne[3] == 1) {
        return ggml_backend_dnnl_inner_product(backend, dst);
    }

    GGML_TENSOR_LOCALS(int64_t, ne_s, src, ne)
    GGML_TENSOR_LOCALS(int64_t, ne_w, weights, ne)
    GGML_TENSOR_LOCALS(int64_t, ne_d, dst, ne)

    GGML_ASSERT(ne_d0 == ne_w1);
    GGML_ASSERT(ne_d1 == ne_s1);
    GGML_ASSERT(ne_d2 == ne_s2);
    GGML_ASSERT(ne_d3 == ne_s3);

    auto dst_mem = ggml_tensor_to_dnnl_mem(backend, dst);
    auto dst_md     = dst_mem.get_desc();
    auto src_mem = ggml_tensor_to_dnnl_mem(backend, src, false, dst_md.get_data_type());
    auto src_md     = src_mem.get_desc();
    auto weights_mem = ggml_tensor_to_dnnl_mem(backend, weights, true, dst_md.get_data_type());
    auto weights_md = weights_mem.get_desc();

    // adjust mems
    auto src_dims = src_md.get_dims();
    auto weights_dims = weights_md.get_dims();
    auto  dst_dims =  dst_md.get_dims();

    // bool adjusted = adjust_for_dnnl_broadcast(src_dims, weights_dims, dst_dims, 2);
    // dnnl::memory wm;
    // dnnl::memory sm;
    // dnnl::memory dm;
    // if (adjusted) {
    //     src_md = src_md.reshape(src_dims);
    //     wm = src_mem;
    //     src_mem = dnnl::memory{src_md, src_mem.get_engine(), src_mem.get_data_handle()};

    //     wm = weights_mem;
    //     weights_md = weights_md.reshape(weights_dims);
    //     weights_mem = dnnl::memory{weights_md, weights_mem.get_engine(), weights_mem.get_data_handle()};

    //     dm = dst_mem;
    //     dst_md = dst_md.reshape(dst_dims);
    //     dst_mem = dnnl::memory(dst_md, dst_mem.get_engine(), dst_mem.get_data_handle());
    // }

    auto pd = dnnl::matmul::primitive_desc{ctx->engine, src_md, weights_md, dst_md};
    auto prim = dnnl::matmul{pd};

    std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC,     src_mem},
        {DNNL_ARG_WEIGHTS, weights_mem},
        {DNNL_ARG_DST,     dst_mem},
    };

    prim.execute(ctx->stream, args);
    ctx->stream.wait();
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_dnnl_cpy(ggml_backend_t backend, struct ggml_tensor * dst) {
    auto * ctx = static_cast<ggml_dnnl_context *>(backend->context);
    auto src = dst->src[0];

    auto src_mem = ggml_tensor_to_dnnl_mem(backend, src);
    auto dst_mem = ggml_tensor_to_dnnl_mem(backend, dst);

    auto src_md = src_mem.get_desc();
    auto dst_md = dst_mem.get_desc();

    auto src_dims = src_md.get_dims();
    auto dst_dims = dst_md.get_dims();

    if (!std::equal(src_dims.begin(), src_dims.end(), dst_dims.begin())) {
        // try reshape src
        auto new_src_md = src_md.reshape(dst_dims, true);
        if (new_src_md) {
            src_md = new_src_md;
            src_mem = dnnl::memory{src_md, ctx->engine, src_mem.get_data_handle()};
        } else {
            // reshape dst
            dst_md = dst_md.reshape(src_dims);
            dst_mem = dnnl::memory{dst_md, ctx->engine, dst_mem.get_data_handle()};
        }
    }

    auto reorder = dnnl::reorder{src_mem, dst_mem};
    reorder.execute(ctx->stream, src_mem, dst_mem);
    ctx->stream.wait();
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_dnnl_scale(ggml_backend_t backend, struct ggml_tensor * dst) {
    auto * ctx = static_cast<ggml_dnnl_context *>(backend->context);
    auto src = dst->src[0];

    auto src_mem = ggml_tensor_to_dnnl_mem(backend, src);
    auto dst_mem = ggml_tensor_to_dnnl_mem(backend, dst);

    float alpha  = *reinterpret_cast<float*>(dst->op_params);

    auto pd = dnnl::eltwise_forward::primitive_desc{ctx->engine, dnnl::prop_kind::forward, dnnl::algorithm::eltwise_linear, src_mem.get_desc(), dst_mem.get_desc(), alpha};
    auto prim = dnnl::eltwise_forward{pd};
    prim.execute(ctx->stream, {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_DST, dst_mem},
    });
    ctx->stream.wait();
    return GGML_STATUS_SUCCESS;
}

// Copy/paste from ggml.c:

static void ggml_backend_dnnl_diag_mask_f32(
        ggml_backend_t backend,
        struct ggml_tensor * dst,
        const float value) {
    GGML_UNUSED(backend);
    const struct ggml_tensor * src = dst->src[0];

    const int  n_past  = ((int32_t *) dst->op_params)[0];

    auto src_mem = ggml_tensor_to_dnnl_mem(backend, src);
    auto dst_mem = ggml_tensor_to_dnnl_mem(backend, dst);

    auto src_ptr = map_memory<char>(src_mem);
    auto dst_ptr = map_memory<char>(dst_mem);

    const bool inplace = src_ptr == dst_ptr;

    GGML_ASSERT(n_past >= 0);

    if (!inplace) {
        // memcpy needs to be synchronized across threads to avoid race conditions.
        // => do it in INIT phase
        GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src));
        GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src));
        std::memcpy(dst_ptr, src_ptr, ggml_nbytes(dst));
    }

    // TODO: handle transposed/permuted matrices

    const int n  = ggml_nrows(src);
    const int nc = src->ne[0];
    const int nr = src->ne[1];
    const int nz = n/nr;

    GGML_ASSERT(dst->nb[0] == sizeof(float));
    GGML_ASSERT(src->nb[0] == sizeof(float));

    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < nr; j += 1) {
            for (int i = n_past; i < nc; i++) {
                if (i > n_past + j) {
                    *(float *)(dst_ptr + k*dst->nb[2] + j*dst->nb[1] + i*dst->nb[0]) = value;
                }
            }
        }
    }
}

static ggml_status ggml_backend_dnnl_diag_mask(
        ggml_backend_t backend,
        struct ggml_tensor * dst, 
        const float value) {

    const struct ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_backend_dnnl_diag_mask_f32(backend, dst, value);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_dnnl_diag_mask_zero(ggml_backend_t b, ggml_tensor* d) {
    return ggml_backend_dnnl_diag_mask(b, d, 0);
}

static ggml_status ggml_backend_dnnl_diag_mask_inf(ggml_backend_t b, ggml_tensor* d) {
    return ggml_backend_dnnl_diag_mask(b, d, -INFINITY);
}

static bool ggml_backend_dnnl_softmax_supported(ggml_backend_t backend, const ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_SOFT_MAX);
    // mask and bias are not supported
    return dst->src[1] == 0
         && dst->op_params[1] == 0
         && unary_tensor_supported(backend, dst);
}

static ggml_status ggml_backend_dnnl_softmax(ggml_backend_t backend, struct ggml_tensor * dst) {
    auto * ctx = static_cast<ggml_dnnl_context *>(backend->context);
    auto src = dst->src[0];

    auto src_mem = ggml_tensor_to_dnnl_mem(backend, src);
    auto dst_mem = ggml_tensor_to_dnnl_mem(backend, dst);

    float input_scale = reinterpret_cast<float*>(dst->op_params)[0];
    //auto scaled_mem = dnnl::memory{src_mem.get_desc(), src_mem.get_engine()};
    // scale input
    if (input_scale != 1.0f) {
        auto pd = dnnl::eltwise_forward::primitive_desc{ctx->engine, dnnl::prop_kind::forward_inference, dnnl::algorithm::eltwise_linear, src_mem.get_desc(), src_mem.get_desc(), input_scale};
        auto prim = dnnl::eltwise_forward{pd};
        prim.execute(ctx->stream, {
            {DNNL_ARG_SRC, src_mem},
            {DNNL_ARG_DST, src_mem},
        });
    }

    const int axis = src_mem.get_desc().get_dims().size() - 1;
    auto pd = dnnl::softmax_forward::primitive_desc{ctx->engine, dnnl::prop_kind::forward_inference, dnnl::algorithm::softmax_accurate, src_mem.get_desc(), dst_mem.get_desc(), axis};
    auto prim = dnnl::softmax_forward{pd};
    prim.execute(ctx->stream, {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_DST, dst_mem},
    });
    ctx->stream.wait();
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_dnnl_norm(ggml_backend_t backend, struct ggml_tensor * dst) {
    auto * ctx = static_cast<ggml_dnnl_context *>(backend->context);
    auto src = dst->src[0];

    auto src_mem = ggml_tensor_to_dnnl_mem(backend, src);
    auto dst_mem = ggml_tensor_to_dnnl_mem(backend, dst);

    float eps = ((const float *)(dst->op_params))[0];

    GGML_ASSERT(eps > 0.0f);

    auto pd = dnnl::layer_normalization_forward::primitive_desc{ctx->engine, dnnl::prop_kind::forward_inference, src_mem.get_desc(), dst_mem.get_desc(), eps, dnnl::normalization_flags::none};
    auto prim = dnnl::layer_normalization_forward{pd};
    prim.execute(ctx->stream, {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_DST, dst_mem},
    });
    ctx->stream.wait();
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_dnnl_binary(ggml_backend_t backend, struct ggml_tensor * dst, dnnl::algorithm op) {
    auto * ctx = static_cast<ggml_dnnl_context *>(backend->context);
    auto src0 = dst->src[0];
    auto src1 = dst->src[1];
    // oneDNN supports in-place for src0 only
    if (src1->data == dst->data) {
        std::swap(src0, src1);
    }

    auto src0_mem = ggml_tensor_to_dnnl_mem(backend, src0);
    auto src1_mem = ggml_tensor_to_dnnl_mem(backend, src1);
    auto dst_mem  = ggml_tensor_to_dnnl_mem(backend, dst);

    // adjust mems
    auto src0_md = src0_mem.get_desc();
    auto src1_md = src1_mem.get_desc();
    auto  dst_md =  dst_mem.get_desc();

    auto src0_dims = src0_md.get_dims();
    auto src1_dims = src1_md.get_dims();
    auto  dst_dims =  dst_md.get_dims();

    bool adjusted = adjust_for_dnnl_broadcast(src0_dims, src1_dims, dst_dims);
    if (adjusted) {
        src0_md = src0_md.reshape(src0_dims);
        src0_mem = dnnl::memory{src0_md, src0_mem.get_engine(), src0_mem.get_data_handle()};

        src1_md = src1_md.reshape(src1_dims);
        src1_mem = dnnl::memory{src1_md, src1_mem.get_engine(), src1_mem.get_data_handle()};

        dst_md = dst_md.reshape(dst_dims);
        dst_mem = dnnl::memory(dst_md, dst_mem.get_engine(), dst_mem.get_data_handle());
    }

    auto pd = dnnl::binary::primitive_desc{ctx->engine, op, src0_md, src1_md, dst_md};
    auto prim = dnnl::binary{pd};
    prim.execute(ctx->stream, {
        {DNNL_ARG_SRC_0, src0_mem},
        {DNNL_ARG_SRC_1, src1_mem},
        {DNNL_ARG_DST,   dst_mem},
    });
    ctx->stream.wait();
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_dnnl_unary(ggml_backend_t backend, struct ggml_tensor * dst, dnnl::algorithm op, float a = 1.0f, float b= 0.0f) {
    auto * ctx = static_cast<ggml_dnnl_context *>(backend->context);
    auto src0 = dst->src[0];

    auto src0_mem = ggml_tensor_to_dnnl_mem(backend, src0);
    auto dst_mem  = ggml_tensor_to_dnnl_mem(backend, dst);

    auto pd = dnnl::eltwise_forward::primitive_desc{ctx->engine, dnnl::prop_kind::forward, op, src0_mem.get_desc(), dst_mem.get_desc(), a, b};
    auto prim = dnnl::eltwise_forward{pd};
    prim.execute(ctx->stream, {
        {DNNL_ARG_SRC_0, src0_mem},
        {DNNL_ARG_DST,   dst_mem},
    });
    ctx->stream.wait();
    return GGML_STATUS_SUCCESS;
}

// ggml_compute_forward_get_rows
static ggml_status ggml_backend_dnnl_get_rows_impl(ggml_backend_t backend, struct ggml_tensor * dst) {
    auto * ctx = static_cast<ggml_dnnl_context *>(backend->context);

    using dims_t = dnnl::memory::dims;

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t nc = ne00;
    const int64_t nr = ggml_nelements(src1);

    GGML_ASSERT(ne0  == nc);
    GGML_ASSERT(ne02 == ne11);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    auto src0_mem = ggml_tensor_to_dnnl_mem(backend, src0);
    auto src1_mem = ggml_tensor_to_dnnl_mem(backend, src1);
    auto dst_mem  = ggml_tensor_to_dnnl_mem(backend, dst);

    // adjust mems
    auto src0_md = src0_mem.get_desc();
    auto src1_md = src1_mem.get_desc();
    auto  dst_md =  dst_mem.get_desc();

    auto src0_dims = src0_md.get_dims();
    auto src1_dims = src1_md.get_dims();
    auto  dst_dims =  dst_md.get_dims();

    auto src1_ptr = map_memory<char>(src1_mem);

    auto row_dims = dims_t{1,1,1,nc};

    for (int64_t i = 0; i < nr; ++i) {
        const int64_t i12 = i/(ne11*ne10);
        const int64_t i11 = (i - i12*ne11*ne10)/ne10;
        const int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        const int64_t i01 = *(int32_t *) (src1_ptr.get() + i10*nb10 + i11*nb11 + i12*nb12);

        auto src_offset = dims_t{i12, i11, i01, 0};
        auto dst_offset = dims_t{i12, i11, i10, 0};
        auto src_sub_md = src0_md.submemory_desc(row_dims, src_offset);
        auto dst_sub_md = dst_md.submemory_desc(row_dims, dst_offset);
        auto src_submem = dnnl::memory{src_sub_md, src0_mem.get_engine(), src0_mem.get_data_handle()};
        auto dst_submem = dnnl::memory{dst_sub_md, dst_mem.get_engine(), dst_mem.get_data_handle()};

        dnnl::reorder prim{src_submem, dst_submem};
        prim.execute(ctx->stream, src_submem, dst_submem);
        ctx->stream.wait();
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_dnnl_get_rows(ggml_backend_t backend, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];

    if (!(ggml_dnnl_tensor_supported(dst) && ggml_dnnl_tensor_supported(src0))) {
        GGML_ASSERT(false);
        return GGML_STATUS_FAILED;
    }

    return ggml_backend_dnnl_get_rows_impl(backend, dst);
}
///////////////////////////

struct dnnl_op_handler {
    bool     (*support_op)(ggml_backend_t, const ggml_tensor*);
    ggml_status (*compute)(ggml_backend_t, ggml_tensor*);
};

#define NOPE_RECORD(op) \
{ op, {\
    [](ggml_backend_t, const ggml_tensor*){ return false; },\
    [](ggml_backend_t, ggml_tensor*){ return GGML_STATUS_SUCCESS; }\
}}

#define UNARY_OP_RECORD(op, alg, alpha, beta) \
{ op,\
    { unary_tensor_supported,\
    [](ggml_backend_t b, ggml_tensor* o) { return ggml_backend_dnnl_unary(b, o, alg, alpha, beta); }}\
}

#define BINARY_OP_RECORD(op, a) \
{ op,\
    { binary_tensor_supported,\
    [](ggml_backend_t b, ggml_tensor* o) { return ggml_backend_dnnl_binary(b, o, a); }}\
}

static const std::unordered_map<ggml_unary_op, dnnl_op_handler>* ggml_dnnl_unary_op_map() {
    using algo = dnnl::algorithm;
    static std::unordered_map<ggml_unary_op, dnnl_op_handler> uop_map_ {
        UNARY_OP_RECORD(GGML_UNARY_OP_ABS, algo::eltwise_abs, 1, 0),
        UNARY_OP_RECORD(GGML_UNARY_OP_TANH, algo::eltwise_tanh, 1, 0),
        UNARY_OP_RECORD(GGML_UNARY_OP_ELU, algo::eltwise_elu, 1, 0),
        UNARY_OP_RECORD(GGML_UNARY_OP_RELU, algo::eltwise_relu, 0, 0),
        UNARY_OP_RECORD(GGML_UNARY_OP_GELU, algo::eltwise_gelu_erf, 1, 0),
        UNARY_OP_RECORD(GGML_UNARY_OP_GELU_QUICK, algo::eltwise_gelu_tanh, 1, 0),
        UNARY_OP_RECORD(GGML_UNARY_OP_HARDSWISH, algo::eltwise_hardswish, 1.f/6.f, 0.5f),
        UNARY_OP_RECORD(GGML_UNARY_OP_HARDSIGMOID, algo::eltwise_hardsigmoid, 1.f/6.f, 0.5f),
        // GGML_UNARY_OP_SILU,
        // GGML_UNARY_OP_SGN,
        // GGML_UNARY_OP_NEG,
        // GGML_UNARY_OP_STEP,
    };
    return &uop_map_;
}

static bool ggml_dnnl_supports_op_unary(ggml_backend_t backend, const ggml_tensor* op) {
    enum ggml_unary_op uop = ggml_get_unary_op(op);
    auto uop_map_ = ggml_dnnl_unary_op_map();
    auto it = uop_map_->find(uop);
    return (it != uop_map_->end()) && it->second.support_op(backend, op);
}

static ggml_status ggml_dnnl_compute_op_unary(ggml_backend_t backend, ggml_tensor* op) {
    enum ggml_unary_op uop = ggml_get_unary_op(op);
    auto uop_map_ = ggml_dnnl_unary_op_map();
    auto it = uop_map_->find(uop);
    if (it == uop_map_->end()) {
        fprintf(stderr, "%s: unsupported op %s\n", __func__, ggml_op_desc(op));
        GGML_ASSERT(false);
        return GGML_STATUS_FAILED;
    }
    return it->second.compute(backend, op);
}

static const std::unordered_map<ggml_op, dnnl_op_handler>* ggml_dnnl_op_map() {
    using algo = dnnl::algorithm;

    static std::unordered_map<ggml_op, dnnl_op_handler> op_map_ = {
        NOPE_RECORD(GGML_OP_NONE),
        NOPE_RECORD(GGML_OP_RESHAPE),
        NOPE_RECORD(GGML_OP_VIEW),
        NOPE_RECORD(GGML_OP_PERMUTE),
        NOPE_RECORD(GGML_OP_TRANSPOSE),

        // UNARY_OP_RECORD(GGML_OP_SQR,  algo::eltwise_square),
        // UNARY_OP_RECORD(GGML_OP_SQRT, algo::eltwise_sqrt),
        // UNARY_OP_RECORD(GGML_OP_LOG,  algo::eltwise_log),

        // BINARY_OP_RECORD(GGML_OP_ADD,  algo::binary_add),
        // BINARY_OP_RECORD(GGML_OP_ADD1, algo::binary_add), // TODO(rfsaliev) use unary
        // BINARY_OP_RECORD(GGML_OP_SUB,  algo::binary_sub),
        // BINARY_OP_RECORD(GGML_OP_MUL,  algo::binary_mul),
        // BINARY_OP_RECORD(GGML_OP_DIV,  algo::binary_div),

        // {GGML_OP_NORM, {unary_tensor_supported, ggml_backend_dnnl_norm }},
        {GGML_OP_MUL_MAT, {ggml_compute_forward_mul_mat_use_dnnl, ggml_backend_dnnl_mul_mat}},

        // {GGML_OP_CONT, {unary_tensor_supported, ggml_backend_dnnl_cpy }},
        // {GGML_OP_CPY,  {unary_tensor_supported, ggml_backend_dnnl_cpy }},
        // {GGML_OP_DUP,  {unary_tensor_supported, ggml_backend_dnnl_cpy }},

        // {GGML_OP_SCALE, {unary_tensor_supported, ggml_backend_dnnl_scale }},
        // {GGML_OP_DIAG_MASK_ZERO, {unary_tensor_supported, ggml_backend_dnnl_diag_mask_zero}},
        // {GGML_OP_DIAG_MASK_INF,  {unary_tensor_supported, ggml_backend_dnnl_diag_mask_inf}},
        // {GGML_OP_SOFT_MAX, {ggml_backend_dnnl_softmax_supported, ggml_backend_dnnl_softmax}},

        // {GGML_OP_GET_ROWS, {unary_tensor_supported, ggml_backend_dnnl_get_rows}},

        // {GGML_OP_UNARY, {ggml_dnnl_supports_op_unary, ggml_dnnl_compute_op_unary}},
    };
    return &op_map_;
}

#undef NOPE_RECORD
#undef UNARY_OP_RECORD
#undef BINARY_OP_RECORD

static ggml_status ggml_backend_dnnl_node_compute(ggml_backend_t backend, struct ggml_tensor * node) {
        auto op_map = ggml_dnnl_op_map();
        auto it = op_map->find(node->op);

        if (it == op_map->end()) {
            fprintf(stderr, "%s: unsupported op %s\n", __func__, ggml_op_desc(node));
            GGML_ASSERT(false);
            return GGML_STATUS_FAILED;
        }
        return it->second.compute(backend, node);
}

static bool ggml_backend_dnnl_node_supported(ggml_backend_t backend, const struct ggml_tensor * node) {
        auto op_map = ggml_dnnl_op_map();
        auto it = op_map->find(node->op);
        return (it != op_map->end()) && it->second.support_op(backend, node);
}

// buffer interface

#if USE_DNNL_BUFFER
// DNNL buffer type

GGML_CALL static const char * ggml_backend_dnnl_buffer_name(ggml_backend_buffer_t buffer) {
    return ggml_backend_dnnl_name_str;

    GGML_UNUSED(buffer);
}

GGML_CALL static void ggml_backend_dnnl_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_dnnl_buffer_context* ctx = (ggml_backend_dnnl_buffer_context*)buffer->context;
    ctx->sub_mems.clear();
    delete ctx;
}

GGML_CALL static void * ggml_backend_dnnl_buffer_get_base(ggml_backend_buffer_t buffer) {
    return (void*)DNNL_BUFFER_BASE;
    GGML_UNUSED(buffer);
}

static void ggml_backend_dnnl_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(buffer == tensor->buffer);

    if (tensor->view_src != NULL && tensor->view_offs == 0) {
        tensor->extra = tensor->view_src->extra;
    } else {
    //printf(" op:%s-'%s'\n", ggml_op_desc(tensor), tensor->name);
        auto buf = tensor->view_src != NULL ? tensor->view_src->buffer : tensor->buffer;
        ggml_backend_dnnl_buffer_context* ctx = (ggml_backend_dnnl_buffer_context*)buf->context;
        dnnl::memory::dim offset = (uintptr_t)tensor->data - DNNL_BUFFER_BASE;
        auto md = ctx->mem.get_desc();
        GGML_ASSERT(md.get_ndims() == 1);
        GGML_ASSERT(md.get_data_type() == dnnl::memory::data_type::s8);

        auto sub_md = md.submemory_desc({(dnnl::memory::dim)ggml_nbytes(tensor)}, {offset});
        auto sub_mem = dnnl::memory{sub_md, ctx->mem.get_engine(), ctx->mem.get_data_handle()};
        ctx->sub_mems.push_back(sub_mem);
        tensor->extra = sub_mem.get();
    }
    tensor->backend = GGML_BACKEND_TYPE_GPU;
    GGML_UNUSED(buffer);
}

GGML_CALL static void ggml_backend_dnnl_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto tensor_mem = dnnl::memory{(dnnl_memory_t)tensor->extra, true};
    auto ptr = map_memory<char>(tensor_mem);
    memcpy(ptr.get() + offset, data, size);
    GGML_UNUSED(buffer);
}

GGML_CALL static void ggml_backend_dnnl_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto tensor_mem = dnnl::memory{(dnnl_memory_t)tensor->extra, true};
    auto ptr = map_memory<char>(tensor_mem);
    memcpy(data, ptr.get() + offset, size);
    GGML_UNUSED(buffer);
}

GGML_CALL static void ggml_backend_dnnl_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_dnnl_buffer_context* ctx = (ggml_backend_dnnl_buffer_context*)buffer->context;
    auto ptr = map_memory(ctx->mem);
    memset(ptr.get(), value, ctx->mem.get_desc().get_size());
}

GGML_CALL static void ggml_backend_dnnl_buffer_reset(ggml_backend_buffer_t buffer) {
    ggml_backend_dnnl_buffer_context* ctx = (ggml_backend_dnnl_buffer_context*)buffer->context;
    ctx->sub_mems.clear();
}

static struct ggml_backend_buffer_i dnnl_backend_buffer_i = {
    /* .get_name        = */ ggml_backend_dnnl_buffer_name,
    /* .free_buffer     = */ ggml_backend_dnnl_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_dnnl_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_dnnl_buffer_init_tensor,
    /* .set_tensor      = */ ggml_backend_dnnl_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_dnnl_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL, //ggml_backend_dnnl_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_dnnl_buffer_clear,
    /* .reset           = */ NULL,
};

// buffer type

GGML_CALL static const char * ggml_backend_dnnl_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return ggml_backend_dnnl_name_str;

    GGML_UNUSED(buft);
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_dnnl_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto* buft_ctx = static_cast<ggml_dnnl_context *>(buft->context);

    dnnl::memory::desc md{{(dnnl::memory::dim)size}, dnnl::memory::data_type::s8, dnnl::memory::dims{}};
    dnnl::memory mem{md, buft_ctx->engine};

    auto buf_ctx = new ggml_backend_dnnl_buffer_context{mem, {}};
    return ggml_backend_buffer_init(buft, dnnl_backend_buffer_i, buf_ctx, size);
}

GGML_CALL static size_t ggml_backend_dnnl_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return DNNL_BUFFER_ALIGNMENT;

    GGML_UNUSED(buft);
}

GGML_CALL static bool ggml_backend_dnnl_buffer_type_supports_backend(ggml_backend_buffer_type_t buft, ggml_backend_t backend) {
    return ggml_backend_is_dnnl(backend); // || ggml_backend_is_cpu(backend);

    GGML_UNUSED(buft);
}

// GGML_CALL static bool ggml_backend_dnnl_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
//     return true;

//     GGML_UNUSED(buft);
// }

GGML_CALL static ggml_backend_buffer_type_t ggml_backend_dnnl_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_dnnl(backend));
    static struct ggml_backend_buffer_type ggml_backend_dnnl_buffer_type = {
        /* .iface = */ {
            /* .get_name         = */ ggml_backend_dnnl_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_dnnl_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_dnnl_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ NULL, // defaults to false // ggml_backend_dnnl_buffer_type_is_host,
        },
        /* .context = */ backend->context,
    };
    return &ggml_backend_dnnl_buffer_type;
}

#else 
// CPU buffer type

// GGML_CALL static const char * ggml_backend_dnnl_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
//     return ggml_backend_dnnl_name_str;

//     GGML_UNUSED(buft);
// }

// GGML_CALL static bool ggml_backend_dnnl_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
//     return true;

//     GGML_UNUSED(buft);
// }

GGML_CALL static ggml_backend_buffer_type_t ggml_backend_dnnl_get_default_buffer_type(ggml_backend_t backend) {
    GGML_UNUSED(backend);
#if 1
    return ggml_backend_cpu_buffer_type();
#else
    auto cpu_buffer_type = ggml_backend_cpu_buffer_type();
    static struct ggml_backend_buffer_type ggml_backend_dnnl_buffer_type = {
        /* .iface = */ {
            /* .get_name         = */ ggml_backend_dnnl_buffer_type_get_name,
            /* .alloc_buffer     = */ cpu_buffer_type->iface.alloc_buffer,  // ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ cpu_buffer_type->iface.get_alignment, // ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_dnnl_buffer_type_is_host,
        },
        /* .context = */ NULL,
    };
    return &ggml_backend_dnnl_buffer_type;
#endif
}
#endif

// backend interface

GGML_CALL static const char * ggml_backend_dnnl_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return ggml_backend_dnnl_name_str;
}

GGML_CALL static void ggml_backend_dnnl_free(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    // GGML_ASSERT(ggml_backend_is_dnnl(backend));
    // auto * ctx = static_cast<ggml_dnnl_context *>(backend->context);

    // if (ctx != nullptr) {
    //     delete ctx;
    // }

    // delete backend;
}

// GGML_CALL static void ggml_backend_dnnl_synchronize(ggml_backend_t backend) {
//     auto * ctx = static_cast<ggml_dnnl_context *>(backend->context);
//     ctx->stream.wait();
// }

GGML_CALL static ggml_status ggml_backend_dnnl_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    //auto * ctx = static_cast<ggml_dnnl_context *>(backend->context);
    GGML_UNUSED(backend);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        //printf(" op:%s-'%s'\n", ggml_op_desc(node), node->name);
        auto status = ggml_backend_dnnl_node_compute(backend, node);
        if (status != GGML_STATUS_SUCCESS) {
            return status;
        }
    }

    return GGML_STATUS_SUCCESS;
}

GGML_CALL static bool ggml_backend_dnnl_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    return ggml_backend_dnnl_node_supported(backend, op);
}

// static bool ggml_backend_buft_is_dnnl(ggml_backend_buffer_type_t buft) {
//     return buft->iface.get_name == ggml_backend_dnnl_buffer_type_get_name;
// }

GGML_CALL static bool ggml_backend_dnnl_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);
// #if USE_DNNL_BUFFER
//     return ggml_backend_buft_is_dnnl(buft);
// #else
//     return /*ggml_backend_buft_is_dnnl(buft); //||*/ ggml_backend_buft_is_host(buft);
// #endif
    GGML_UNUSED(backend);
}


static struct ggml_backend_i dnnl_backend_i = {
    /* .get_name                = */ ggml_backend_dnnl_name,
    /* .free                    = */ ggml_backend_dnnl_free,
    /* .get_default_buffer_type = */ ggml_backend_dnnl_get_default_buffer_type,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL, //ggml_backend_dnnl_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_dnnl_graph_compute,
    /* .supports_op             = */ ggml_backend_dnnl_supports_op,
    /* .supports_buft           = */ ggml_backend_dnnl_supports_buft,
    /* .offload_op              = */ NULL,
    /* .event_new               = */ NULL,
    /* .event_free              = */ NULL,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .event_synchronize       = */ NULL
};

static ggml_guid_t ggml_backend_dnnl_guid() {
    // ce8e0c82-fe2f-11ee-9f1a-efb05da0c44a
    static ggml_guid guid = {0xce, 0x8e, 0x0c, 0x82, 0xfe, 0x2f, 0x11, 0xee, 0x9f, 0x1a, 0xef, 0xb0, 0x5d, 0xa0, 0xc4, 0x4a};
    return &guid;
}

GGML_CALL ggml_backend_t ggml_backend_dnnl_init() {
    // GGML_ASSERT(s_dnnl_context == nullptr);
    // s_dnnl_context = new ggml_dnnl_context(device);

    static ggml_dnnl_context dnnl_context_{0};

    ggml_backend_t dnnl_backend = new ggml_backend {
        /* .guid      = */ ggml_backend_dnnl_guid(),
        /* .interface = */ dnnl_backend_i,
        /* .context   = */ &dnnl_context_,
    };

    return dnnl_backend;
}

GGML_CALL bool ggml_backend_is_dnnl(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_dnnl_guid());
}

// backend registry
GGML_CALL static ggml_backend_t ggml_backend_reg_dnnl_init(const char * params, void * user_data) {
    ggml_backend_t dnnl_backend = ggml_backend_dnnl_init();
    return dnnl_backend;

    GGML_UNUSED(params);
    GGML_UNUSED(user_data);
}

extern "C" GGML_CALL int ggml_backend_dnnl_reg_devices(void);

GGML_CALL int ggml_backend_dnnl_reg_devices() {
    ggml_backend_register(ggml_backend_dnnl_name_str, ggml_backend_reg_dnnl_init, ggml_backend_dnnl_get_default_buffer_type(ggml_backend_dnnl_init()), NULL);
    int device_count = 1;
    return device_count;
}
