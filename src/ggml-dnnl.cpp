#include "ggml.h"
#include "ggml-dnnl.h"
#include "ggml-backend-impl.h"

#include <utility>

#include <dnnl.hpp>

struct ggml_dnnl_context {
    dnnl::engine engine;

    ggml_dnnl_context(int device) : engine(dnnl::engine::kind::cpu, device) {}
};

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

static bool ggml_compute_forward_mul_mat_use_dnnl(const struct ggml_tensor * dst) {
#if 0
    return true;
    GGML_UNUSED(dst);
#else
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    if (true
        && dst->op != GGML_OP_MUL_MAT_ID
        && ggml_dnnl_tensor_supported(src0)
        && ggml_dnnl_tensor_supported(src1)
        && ggml_dnnl_tensor_supported(dst)
        && ggml_is_contiguous(src0)
        && ggml_is_contiguous(src1)
        // && (ne0 >= 32 && ne1 >= 32 && ne10 >= 32)
        ) {

        /*printf("BLAS: %d %d %d %d %d\n", ne0, ne1, ne10, ne00, ne01);*/
        return true;
    }

    return false;
#endif
}

dnnl::memory::desc ggml_tensor_to_dnnl_md(const struct ggml_tensor * t, bool transpose = false) {
    GGML_ASSERT(ggml_dnnl_tensor_supported(t));
    using dim = dnnl::memory::dim;
    using dims = dnnl::memory::dims;

    GGML_TENSOR_LOCALS(int64_t, ne,  t,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  t,  nb)

    auto adims = dims{ne3, ne2, (transpose ? ne0 : ne1), (transpose ? ne1 : ne0)};
    auto dt = ggml_type_to_dnnl_dtype(t->type);
    auto strides = dims{
                        (dim)(nb3/nb0),
                        (dim)(nb2/nb0),
                        (dim)((transpose ? nb0 : nb1)/nb0),
                        (dim)((transpose ? nb1 : nb0)/nb0)
                  };
    return dnnl::memory::desc{adims, dt, strides};
}

static void ggml_backend_dnnl_mul_mat(ggml_backend_t backend, struct ggml_tensor * dst) {
    auto * ctx = static_cast<ggml_dnnl_context *>(backend->context);
    auto& engine = ctx->engine;

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // NOTE: src0 - weights, src1 - data
    auto src0_md = ggml_tensor_to_dnnl_md(src0, true);
    auto src1_md = ggml_tensor_to_dnnl_md(src1);
    auto dst_md = ggml_tensor_to_dnnl_md(dst);

    auto pd = dnnl::matmul::primitive_desc{engine, src1_md, src0_md, dst_md};
    auto prim = dnnl::matmul{pd};

    auto stream = dnnl::stream{engine};

    // NOTE: src0 - weights, src1 - data
    std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_WEIGHTS, dnnl::memory{src0_md, engine, src0->data}},
        {DNNL_ARG_SRC, dnnl::memory{src1_md, engine, src1->data}},
        {DNNL_ARG_DST, dnnl::memory{dst_md, engine, dst->data}},
    };

    prim.execute(stream, args);

    stream.wait();
}

// backend interface

static const char * ggml_backend_dnnl_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "DNNL";
}

static void ggml_backend_dnnl_free(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_dnnl(backend));
    auto * ctx = static_cast<ggml_dnnl_context *>(backend->context);

    if (ctx != nullptr) {
        delete ctx;
    }

    delete backend;
}

static ggml_backend_buffer_type_t ggml_backend_dnnl_get_default_buffer_type(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return ggml_backend_cpu_buffer_type();
}

static ggml_status ggml_backend_dnnl_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    //auto * ctx = static_cast<ggml_dnnl_context *>(backend->context);
    GGML_UNUSED(backend);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_backend_dnnl_mul_mat(backend, node);
                break;

            // TODO
            //case GGML_OP_OUT_PROD:

            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
                fprintf(stderr, "%s: unsupported op %s\n", __func__, ggml_op_desc(node));
                GGML_ASSERT(false);
        }
    }

    return GGML_STATUS_SUCCESS;
}

static bool ggml_backend_dnnl_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_UNUSED(backend);
    return op->op == GGML_OP_MUL_MAT && ggml_compute_forward_mul_mat_use_dnnl(op);
}

static struct ggml_backend_i dnnl_backend_i = {
    /* .get_name                = */ ggml_backend_dnnl_name,
    /* .free                    = */ ggml_backend_dnnl_free,
    /* .get_default_buffer_type = */ ggml_backend_dnnl_get_default_buffer_type,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_dnnl_graph_compute,
    /* .supports_op             = */ ggml_backend_dnnl_supports_op,
    /* .offload_op              = */ NULL,
    /* .event_new               = */ NULL,
    /* .event_free              = */ NULL,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .event_synchronize       = */ NULL,
};

static ggml_guid_t ggml_backend_dnnl_guid() {
    // ce8e0c82-fe2f-11ee-9f1a-efb05da0c44a
    static ggml_guid guid = {0xce, 0x8e, 0x0c, 0x82, 0xfe, 0x2f, 0x11, 0xee, 0x9f, 0x1a, 0xef, 0xb0, 0x5d, 0xa0, 0xc4, 0x4a};
    return &guid;
}

GGML_CALL ggml_backend_t ggml_backend_dnnl_init() {
    // GGML_ASSERT(s_dnnl_context == nullptr);
    // s_dnnl_context = new ggml_dnnl_context(device);

    ggml_dnnl_context* dnnl_context = new ggml_dnnl_context(0);

    ggml_backend_t dnnl_backend = new ggml_backend {
        /* .guid      = */ ggml_backend_dnnl_guid(),
        /* .interface = */ dnnl_backend_i,
        /* .context   = */ dnnl_context,
    };

    return dnnl_backend;
}

GGML_CALL bool ggml_backend_is_dnnl(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_dnnl_guid());
}
