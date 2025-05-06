import numpy as np
from egglog import (
    Expr,
    StringLike,
    Vec,
    function,
    i64,
    panic,
    rewrite,
    rule,
    ruleset,
    set_,
    subsume,
    union,
)
from sealir.eqsat.py_eqsat import (
    Py_AddIO,
    Py_AttrIO,
    Py_Call,
    Py_DivIO,
    Py_LoadGlobal,
    Py_MulIO,
    Py_PowIO,
)
from sealir.eqsat.rvsdg_eqsat import (
    Term,
    TermList,
)

from ch04_1_typeinfer_ifelse import (
    ExtendEGraphToRVSDG as ch04_1_ExtendEGraphToRVSDG,
)
from ch04_1_typeinfer_ifelse import (
    Grammar,
    NbOp_Base,
    String,
    Type,
    TypeFloat64,
    TypeInt64,
    TypeVar,
    make_rule_for_binop,
    setup_argtypes,
)
from ch05_typeinfer_array import (
    Backend,
    ExtendEGraphToRVSDG,
    MyCostModel,
    NbOp_Type,
    base_ruleset,
    compiler_pipeline,
)


class Module(Expr):
    def __init__(self, name: StringLike): ...

    def toType(self) -> Type: ...


@function
def ModuleGetAttr(mod: Module, attrname: StringLike) -> Term: ...


@ruleset
def facts_numpy_module(io: Term, name: String, op: Term, args: Vec[Term]):

    yield rule(
        op == Py_LoadGlobal(io, name),
        name == String("np"),
    ).then(set_(TypeVar(op).getType()).to(Module("numpy").toType()))

    # ------ attributes ------
    numpy_mod = Module("numpy")

    def unary_func(fname, target_func):
        return rule(
            op
            == (
                stmt := Py_Call(
                    func=ModuleGetAttr(numpy_mod, fname),
                    io=io,
                    args=TermList(args),
                )
            ),
            args.length() == i64(1),
        ).then(
            subsume(stmt),
            union(op.getPort(0)).with_(io),
            union(op.getPort(1)).with_(target_func(args[0])),
        )

    # np.pi
    const_pi = ModuleGetAttr(numpy_mod, "pi")
    yield rewrite(
        const_pi,
        subsume=True,
    ).to(Term.LiteralF64(np.pi))
    # np.float32
    yield unary_func("float32", Npy_float32)
    # np.sqrt
    yield unary_func("sqrt", Npy_sqrt)
    # np.tanh
    yield unary_func("tanh", Npy_tanh)


@function
def Npy_float32(val: Term) -> Term: ...
@function
def Npy_sqrt(val: Term) -> Term: ...
@function
def Npy_tanh(val: Term) -> Term: ...


@function
def Npy_float32_from_float64(val: Term) -> Term: ...
@function
def Npy_sqrt_float32(val: Term) -> Term: ...
@function
def Npy_tanh_float32(val: Term) -> Term: ...


@ruleset
def ruleset_typeinfer_numpy_functions(res: Term, arg: Term):
    # float32()
    yield rule(
        res == Npy_float32(arg),
    ).then(set_(TypeVar(res).getType()).to(TypeFloat32))
    yield rewrite(Npy_float32(arg), subsume=True).to(
        Npy_float32_from_float64(arg),
        TypeVar(arg).getType() == TypeFloat64,
    )

    # others

    for func, typed_func in [
        (Npy_sqrt, Npy_sqrt_float32),
        (Npy_tanh, Npy_tanh_float32),
    ]:
        yield rewrite(func(arg), subsume=True).to(
            typed_func(arg),
            TypeVar(arg).getType() == TypeFloat32,
        )
        yield rule(
            res == typed_func(arg),
        ).then(set_(TypeVar(res).getType()).to(TypeFloat32))


@ruleset
def ruleset_module(
    io: Term, name: String, modname: String, op: Term, obj: Term
):
    # Getattribute
    yield rule(
        op == Py_AttrIO(io, obj, name),
        TypeVar(obj).getType() == Module(modname).toType(),
    ).then(
        # Shortcut io
        union(op.getPort(0)).with_(io),
        # Setup getattr
        union(op.getPort(1)).with_(ModuleGetAttr(Module(modname), name)),
    )


@function
def Nb_Add_Float32(lhs: Term, rhs: Term) -> Term: ...


@function
def Nb_Mul_Float32(lhs: Term, rhs: Term) -> Term: ...


@function
def Nb_Div_Float32(lhs: Term, rhs: Term) -> Term: ...


@function
def Nb_Pow_Float32_Int64(lhs: Term, rhs: Term) -> Term: ...


@ruleset
def ruleset_typeinfer_f32_ops():
    yield make_rule_for_binop(
        Py_AddIO, TypeFloat32, TypeFloat32, Nb_Add_Float32, TypeFloat32
    )
    yield make_rule_for_binop(
        Py_MulIO, TypeFloat32, TypeFloat32, Nb_Mul_Float32, TypeFloat32
    )
    yield make_rule_for_binop(
        Py_DivIO, TypeFloat32, TypeFloat32, Nb_Div_Float32, TypeFloat32
    )
    yield make_rule_for_binop(
        Py_PowIO, TypeFloat32, TypeInt64, Nb_Pow_Float32_Int64, TypeFloat32
    )


additional_rules = (
    facts_numpy_module
    | ruleset_module
    | ruleset_typeinfer_numpy_functions
    | ruleset_typeinfer_f32_ops
)


def geglu_tanh_forward(a):
    dt = np.float32
    result = (
        dt(0.5)
        * a
        * (
            dt(1)
            + np.tanh(np.sqrt(dt(2) / dt(np.pi)) * (a + dt(0.044715) * a**3))
        )
    )
    return result


TypeFloat32 = Type.simple("Float32")

Float32 = NbOp_Type("Float32")

from sealir import rvsdg

SExpr = rvsdg.grammar.SExpr


class NpyOp_Float32(NbOp_Base):
    operand: SExpr


class NpyOp_Sqrt_Float32(NbOp_Base):
    operand: SExpr


class NpyOp_Tanh_Float32(NbOp_Base):
    operand: SExpr


class NbOp_Mul_Float32(NbOp_Base):
    lhs: SExpr
    rhs: SExpr


class NbOp_Div_Float32(NbOp_Base):
    lhs: SExpr
    rhs: SExpr


class NbOp_Add_Float32(NbOp_Base):
    lhs: SExpr
    rhs: SExpr


class NbOp_Pow_Float32_Int64(NbOp_Base):
    lhs: SExpr
    rhs: SExpr


class NbOp_module(NbOp_Base):
    name: str


from sealir.rvsdg import grammar as rg


class ExtendEGraphToRVSDG(ch04_1_ExtendEGraphToRVSDG):

    def handle_Term(self, op: str, children: dict | list, grm: Grammar):

        match op, children:
            case "Py_Float", {"val": float(arg)}:
                return grm.write(rg.PyFloat(arg))
            case "Npy_float32", {"val": expr}:
                return grm.write(NpyOp_Float32(expr))
            case "Nb_Mul_Float32", {"lhs": lhs, "rhs": rhs}:
                return grm.write(NbOp_Mul_Float32(lhs=lhs, rhs=rhs))
            case "Nb_Add_Float32", {"lhs": lhs, "rhs": rhs}:
                return grm.write(NbOp_Add_Float32(lhs=lhs, rhs=rhs))
            case "Nb_Div_Float32", {"lhs": lhs, "rhs": rhs}:
                return grm.write(NbOp_Div_Float32(lhs=lhs, rhs=rhs))
            case "Nb_Pow_Float32_Int64", {"lhs": lhs, "rhs": rhs}:
                return grm.write(NbOp_Pow_Float32_Int64(lhs=lhs, rhs=rhs))
            case "Npy_sqrt_float32", {"val": val}:
                return grm.write(NpyOp_Sqrt_Float32(val))
            case "Npy_tanh_float32", {"val": val}:
                return grm.write(NpyOp_Tanh_Float32(val))
            # ---
            case "ModuleGetAttr", {"mod": mod, "attrname": str(attrname)}:
                return grm.write(rg.Undef(str(op)))
            case _:
                # Use parent's implementation for other terms.
                return super().handle_Term(op, children, grm)

    def handle_Module(
        self, key: str, op: str, children: dict | list, grm: Grammar
    ):
        return grm.write(rg.Undef(str(key)))


jt = compiler_pipeline(
    geglu_tanh_forward,
    argtypes=(Float32,),
    ruleset=(base_ruleset | setup_argtypes(TypeFloat32) | additional_rules),
    verbose=True,
    converter_class=ExtendEGraphToRVSDG,
    cost_model=MyCostModel(),
    backend=Backend(),
)
