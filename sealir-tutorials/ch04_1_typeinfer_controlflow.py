# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: sealir_basic_compiler
#     language: python
#     name: python3
# ---

# ## Ch 4 Part 1. Fully typing a scalar function with if-else branch
#
# We will consider control-flow constructs---the if-else branch. We went from
# per-operation inference, to considering the entire function.


from __future__ import annotations

import ctypes
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from egglog import (
    EGraph,
    Expr,
    String,
    StringLike,
    Vec,
    function,
    i64,
    i64Like,
    method,
    rewrite,
    rule,
    ruleset,
    set_,
    union,
)
from llvmlite import binding as llvm
from llvmlite import ir
from sealir import ase, grammar, rvsdg
from sealir.eqsat.py_eqsat import Py_AddIO, Py_GtIO, Py_LtIO, Py_SubIO
from sealir.eqsat.rvsdg_eqsat import (
    GraphRoot,
    InPorts,
    Port,
    PortList,
    Region,
    Term,
    TermList,
    i64,
    wildcard,
)
from sealir.eqsat.rvsdg_extract import (
    CostModel,
    EGraphToRVSDG,
)
from sealir.rvsdg import grammar as rg

from ch03_egraph_program_rewrites import (
    frontend,
    run_test,
)
from ch04_0_typeinfer_scalar import (
    basic_ruleset,
    middle_end,
)
from utils import IN_NOTEBOOK


def compiler_pipeline(
    fn,
    argtypes,
    *,
    verbose=False,
    ruleset,
    converter_class=EGraphToRVSDG,
    cost_model=None,
    backend,
):
    rvsdg_expr, dbginfo = frontend(fn)

    # Middle end
    def define_egraph(egraph: EGraph, func):
        root = GraphRoot(func)
        egraph.let("root", root)

        egraph.run(ruleset.saturate())
        if verbose and IN_NOTEBOOK:
            # For inspecting the egraph
            egraph.display(graphviz=True)
        # egraph.display()

    cost, extracted = middle_end(
        rvsdg_expr,
        define_egraph,
        converter_class=converter_class,
        cost_model=cost_model,
    )
    print("Extracted from EGraph".center(80, "="))
    print("cost =", cost)
    print(rvsdg.format_rvsdg(extracted))

    llmod = backend.lower(extracted, argtypes)
    if verbose:
        print("LLVM module".center(80, "="))
        print(llmod)
    return backend.jit_compile(llmod, extracted)


_wc = wildcard


class Type(Expr):
    @classmethod
    def simple(self, name: StringLike) -> Type: ...

    def __or__(self, other: Type) -> Type: ...


class TypeVar(Expr):
    def __init__(self, term: Term): ...

    @method(merge=lambda x, y: x | y)
    def getType(self) -> Type: ...


@function
def Nb_Gt_Int64(lhs: Term, rhs: Term) -> Term: ...


@function
def Nb_Lt_Int64(lhs: Term, rhs: Term) -> Term: ...


@function
def Nb_Add_Int64(lhs: Term, rhs: Term) -> Term: ...


@function
def Nb_Sub_Int64(lhs: Term, rhs: Term) -> Term: ...


SExpr = rvsdg.grammar.SExpr


class NbOp_Base(grammar.Rule):
    pass


class NbOp_Type(NbOp_Base):
    name: str


class NbOp_InTypeAttr(NbOp_Base):
    idx: int
    type: NbOp_Type


class NbOp_OutTypeAttr(NbOp_Base):
    idx: int
    type: NbOp_Type


class NbOp_Gt_Int64(NbOp_Base):
    lhs: SExpr
    rhs: SExpr


class NbOp_Lt_Int64(NbOp_Base):
    lhs: SExpr
    rhs: SExpr


class NbOp_Add_Int64(NbOp_Base):
    lhs: SExpr
    rhs: SExpr


class NbOp_Sub_Int64(NbOp_Base):
    lhs: SExpr
    rhs: SExpr


class NbOp_Not_Int64(NbOp_Base):
    operand: SExpr


class NbOp_Undef_Int64(NbOp_Base): ...


class Grammar(grammar.Grammar):
    start = rvsdg.Grammar.start | NbOp_Base


from sealir.rvsdg import restructuring


def my_attr_format(attrs: rg.Attrs) -> str:
    ins = {}
    outs = {}
    for attr in attrs.attrs:
        match attr:
            case NbOp_InTypeAttr(idx=int(idx), type=NbOp_Type(name=str(name))):
                ins[idx] = name
            case NbOp_OutTypeAttr(
                idx=int(idx), type=NbOp_Type(name=str(name))
            ):
                outs[idx] = name
            case _:
                assert False

    def format(dct):
        if len(dct):
            hi = max(dct.keys())
            out = ", ".join(dct.get(i, "_") for i in range(hi + 1))
            return f"({out})"
        else:
            return "()"

    return format(ins) + "->" + format(outs)


restructuring.format_attributes = my_attr_format


class ExtendEGraphToRVSDG(EGraphToRVSDG):
    grammar = Grammar

    def handle_region_attributes(self, key: str, grm: Grammar):

        def search_equiv_calls(self_key: str):
            nodes = self.gdct["nodes"]
            ecl = nodes[self_key]["eclass"]
            for k, v in nodes.items():
                children = v["children"]
                if children and nodes[children[0]]["eclass"] == ecl:
                    yield k, v

        def get_types(key_arg):
            typs = []
            for k, v in search_equiv_calls(key_arg):
                for j in self.search_eclass_siblings(k):
                    op = self.gdct["nodes"][j]["op"]
                    if op.startswith("Type."):
                        typ = self.dispatch(j, grm)
                        typs.append(typ)
            return typs

        attrs = []
        typedargs = list(self.search_calls(key, "TypedIns"))
        if typedargs:
            [typedarg] = typedargs
            for key_arg in self.search_method_calls(typedarg, "arg"):
                _k_self, k_idx = self.get_children(key_arg)
                # get the idx in `.arg(idx)`
                idx = self.dispatch(k_idx, grm)
                typs = get_types(key_arg)

                if len(typs) == 1:
                    typ = typs[0]
                    attrs.append(grm.write(NbOp_InTypeAttr(idx=idx, type=typ)))
                else:
                    resolved = list(map(ase.pretty_str, typs))
                    assert len(typs) == 0, f"multiple types: {resolved}"

        typedouts = list(self.search_calls(key, "TypedOuts"))
        if typedouts:
            [typedout] = typedouts
            for key_at in self.search_method_calls(typedout, "at"):
                _k_self, k_idx = self.get_children(key_at)
                idx = self.dispatch(k_idx, grm)

                typs = get_types(key_at)
                if len(typs) == 1:
                    typ = typs[0]
                    attrs.append(
                        grm.write(NbOp_OutTypeAttr(idx=idx, type=typ))
                    )
                else:
                    assert len(typs) == 0, "multiple types"

        return grm.write(rg.Attrs(tuple(attrs)))

    def handle_Type(
        self, key: str, op: str, children: dict | list, grm: Grammar
    ):
        assert op == "Type.simple"
        match children:
            case {"name": name}:
                return grm.write(NbOp_Type(name))
        raise NotImplementedError

    def handle_Term(self, op: str, children: dict | list, grm: Grammar):
        match op, children:

            case "Nb_Gt_Int64", {"lhs": lhs, "rhs": rhs}:
                return grm.write(NbOp_Gt_Int64(lhs=lhs, rhs=rhs))
            case "Nb_Lt_Int64", {"lhs": lhs, "rhs": rhs}:
                return grm.write(NbOp_Lt_Int64(lhs=lhs, rhs=rhs))
            case "Nb_Add_Int64", {"lhs": lhs, "rhs": rhs}:
                return grm.write(NbOp_Add_Int64(lhs=lhs, rhs=rhs))
            case "Nb_Sub_Int64", {"lhs": lhs, "rhs": rhs}:
                return grm.write(NbOp_Sub_Int64(lhs=lhs, rhs=rhs))
            case "Nb_Not_Int64", {"operand": operand}:
                return grm.write(NbOp_Not_Int64(operand=operand))
            case "Nb_Undef_Int64", {}:
                return grm.write(NbOp_Undef_Int64())
            case _:
                # Use parent's implementation for other terms.
                return super().handle_Term(op, children, grm)


class MyCostModel(CostModel):
    def get_cost_function(self, nodename, op, ty, cost, nodes, child_costs):
        self_cost = None

        match op:
            case (
                "Nb_Add_Int64"
                | "Nb_Sub_Int64"
                | "Nb_Gt_Int64"
                | "Nb_Lt_Int64"
                | "Nb_Not_Int64"
            ):
                self_cost = 0.1

        if self_cost is not None:
            return self_cost + sum(child_costs)

        # Fallthrough to parent's cost function
        return super().get_cost_function(
            nodename, op, ty, cost, nodes, child_costs
        )


TypeInt64 = Type.simple("Int64")
TypeBool = Type.simple("Bool")


@ruleset
def ruleset_type_infer_gt(io: Term, x: Term, y: Term, add: Term):
    yield rule(
        add == Py_GtIO(io, x, y),
        TypeVar(x).getType() == TypeInt64,
        TypeVar(y).getType() == TypeInt64,
    ).then(
        # convert to a typed operation
        union(add.getPort(1)).with_(Nb_Gt_Int64(x, y)),
        # shortcut io
        union(add.getPort(0)).with_(io),
        # output type
        set_(TypeVar(add.getPort(1)).getType()).to(TypeBool),
    )


@ruleset
def ruleset_type_infer_lt(io: Term, x: Term, y: Term, add: Term):
    yield rule(
        add == Py_LtIO(io, x, y),
        TypeVar(x).getType() == TypeInt64,
        TypeVar(y).getType() == TypeInt64,
    ).then(
        # convert to a typed operation
        union(add.getPort(1)).with_(Nb_Lt_Int64(x, y)),
        # shortcut io
        union(add.getPort(0)).with_(io),
        # output type
        set_(TypeVar(add.getPort(1)).getType()).to(TypeBool),
    )


@ruleset
def ruleset_type_infer_add(io: Term, x: Term, y: Term, add: Term):
    yield rule(
        add == Py_AddIO(io, x, y),
        TypeVar(x).getType() == TypeInt64,
        TypeVar(y).getType() == TypeInt64,
    ).then(
        # convert to a typed operation
        union(add.getPort(1)).with_(Nb_Add_Int64(x, y)),
        # shortcut io
        union(add.getPort(0)).with_(io),
        # output type
        set_(TypeVar(add.getPort(1)).getType()).to(TypeInt64),
    )


@ruleset
def ruleset_type_infer_sub(io: Term, x: Term, y: Term, add: Term):
    yield rule(
        add == Py_SubIO(io, x, y),
        TypeVar(x).getType() == TypeInt64,
        TypeVar(y).getType() == TypeInt64,
    ).then(
        # convert to a typed operation
        union(add.getPort(1)).with_(Nb_Sub_Int64(x, y)),
        # shortcut io
        union(add.getPort(0)).with_(io),
        # output type
        set_(TypeVar(add.getPort(1)).getType()).to(TypeInt64),
    )


@ruleset
def ruleset_type_unify(ta: Type, tb: Type):
    yield rewrite(ta | tb, subsume=True).to(ta, ta == tb)
    yield rewrite(Type.simple("undef") | tb, subsume=True).to(tb)
    yield rewrite(ta | Type.simple("undef"), subsume=True).to(ta)


@ruleset
def ruleset_propagate_typeof_ifelse(
    then_region: Region,
    else_region: Region,
    idx: i64,
    ifelse: Term,
    then_ports: PortList,
    else_ports: PortList,
    operands: Vec[Term],
):

    yield rule(
        # Propagate operand types into the contained regions
        Term.IfElse(
            cond=_wc(Term),
            then=Term.RegionEnd(region=then_region, ports=_wc(PortList)),
            orelse=Term.RegionEnd(region=else_region, ports=_wc(PortList)),
            operands=TermList(operands),
        ),
        then_region.get(idx),
    ).then(
        union(TypeVar(operands[idx])).with_(TypedIns(then_region).arg(idx)),
        union(TypeVar(operands[idx])).with_(TypedIns(else_region).arg(idx)),
    )

    yield rule(
        # Propagate output types from the contained regions
        ifelse
        == Term.IfElse(
            cond=_wc(Term),
            then=Term.RegionEnd(region=_wc(Region), ports=then_ports),
            orelse=Term.RegionEnd(region=_wc(Region), ports=else_ports),
            operands=TermList(operands),
        ),
        ta := TypeVar(then_ports.getValue(idx)),
        tb := TypeVar(else_ports.getValue(idx)),
    ).then(
        union(ta).with_(tb),
        union(ta).with_(TypeVar(ifelse.getPort(idx))),
    )


class TypedIns(Expr):
    def __init__(self, region: Region): ...

    def arg(self, idx: i64Like) -> TypeVar: ...


class TypedOuts(Expr):
    def __init__(self, region: Region): ...

    def at(self, idx: i64Like) -> TypeVar: ...


@ruleset
def ruleset_region_types(
    region: Region,
    idx: i64,
    typ: TypeVar,
    term: Term,
    portlist: PortList,
):
    yield rule(
        # Inputs
        typ == TypedIns(region).arg(idx),
        term == region.get(idx),
    ).then(
        union(TypeVar(term)).with_(typ),
    )

    yield rule(
        # Outputs
        term == Term.RegionEnd(region=region, ports=portlist),
        pv := portlist.getValue(idx),
    ).then(
        union(TypedOuts(region).at(idx)).with_(TypeVar(pv)),
    )


@ruleset
def facts_function_types(
    outports: Vec[Port],
    func_uid: String,
    reg_uid: String,
    fname: String,
    region: Region,
):
    yield rule(
        GraphRoot(
            Term.Func(
                body=Term.RegionEnd(region=region, ports=PortList(outports)),
                uid=func_uid,
                fname=fname,
            )
        ),
        region == Region(uid=reg_uid, inports=_wc(InPorts)),
    ).then(
        set_(TypedIns(region).arg(1).getType()).to(TypeInt64),
        set_(TypedIns(region).arg(2).getType()).to(TypeInt64),
    )


# Lower
def get_port_by_name(ports: Sequence[rg.Port], name: str):
    for i, p in enumerate(ports):
        if p.name == name:
            return i, p
    raise ValueError(f"{name!r} not found")


class Attributes:
    _typedins: tuple[NbOp_InTypeAttr, ...]
    _typedouts: tuple[NbOp_OutTypeAttr, ...]

    def __init__(self, attrs: rg.Attrs):
        self._typedins = tuple(
            filter(lambda x: isinstance(x, NbOp_InTypeAttr), attrs.attrs)
        )
        self._typedouts = tuple(
            filter(lambda x: isinstance(x, NbOp_OutTypeAttr), attrs.attrs)
        )

    def get_typed_out(self, idx: int):
        for at in self._typedouts:
            if at.idx == idx:
                return at
        raise IndexError(idx)

    def get_return_type(self, regionend: rg.RegionEnd):
        i, p = get_port_by_name(regionend.ports, rvsdg.internal_prefix("ret"))
        return self.get_typed_out(i).type


class Backend:
    def __init__(self):
        self.initialize_llvm()

    def initialize_llvm(self):
        llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()

    def lower_type(self, ty: NbOp_Type):
        match ty:
            case NbOp_Type("Int64"):
                return ir.IntType(64)
            case NbOp_Type("Bool"):
                return ir.IntType(1)
        raise NotImplementedError(f"unknown type: {ty}")

    def lower_io_type(self):
        # IO type is an empty struct
        return ir.LiteralStructType(())

    def lower(self, root: rg.Func, argtypes):
        mod = ir.Module()
        llargtypes = [*map(self.lower_type, argtypes)]

        fname = root.fname
        retty = Attributes(root.body.begin.attrs).get_return_type(root.body)
        llrettype = self.lower_type(retty)

        fnty = ir.FunctionType(llrettype, llargtypes)
        fn = ir.Function(mod, fnty, name=fname)
        # init entry block and builder
        builder = ir.IRBuilder(fn.append_basic_block("entry"))
        iostate = ir.LiteralStructType(())(ir.Undefined)

        # Emit the function body
        reg_args_stack = []

        @contextmanager
        def push(*regionargs):
            reg_args_stack.append(regionargs)
            yield
            reg_args_stack.pop()

        def tos():
            return reg_args_stack[-1]

        def lower_expr(expr: SExpr, state: ase.TraverseState):
            match expr:
                case rg.RegionBegin(inports=inports):
                    values = tos()
                    assert len(values) == len(inports)
                    return values
                case rg.RegionEnd(begin=begin, ports=ports):
                    yield begin
                    portvalues = []
                    for p in ports:
                        pv = yield p.value
                        portvalues.append(pv)
                    return portvalues

                case rg.IfElse(
                    cond=cond, body=body, orelse=orelse, operands=operands
                ):
                    condval = yield cond

                    # process operands
                    ops = []
                    for op in operands:
                        ops.append((yield op))

                    # unpack pybool
                    match condval.type:
                        case ir.IntType() if condval.type.width == 1:
                            condbit = condval
                        case _:
                            raise NotImplementedError(
                                f"unhandled if-cond type: {condval.type}"
                            )

                    bb_then = builder.append_basic_block("then")
                    bb_else = builder.append_basic_block("else")
                    bb_endif = builder.append_basic_block("endif")

                    builder.cbranch(condbit, bb_then, bb_else)
                    # Then
                    with builder.goto_block(bb_then):
                        with push(*ops):
                            value_then = yield body
                        builder.branch(bb_endif)
                        bb_then_end = builder.basic_block
                    # Else
                    with builder.goto_block(bb_else):
                        with push(*ops):
                            value_else = yield orelse
                        builder.branch(bb_endif)
                        bb_else_end = builder.basic_block
                    # EndIf
                    builder.position_at_end(bb_endif)
                    assert len(value_then) == len(value_else)
                    phis = []
                    for i, (left, right) in enumerate(
                        zip(value_then, value_else, strict=True)
                    ):
                        assert left.type == right.type
                        phi = builder.phi(left.type, name=f"ifelse_{i}")
                        phi.add_incoming(left, bb_then_end)
                        phi.add_incoming(right, bb_else_end)
                        phis.append(phi)
                    return phis

                case rg.Unpack(val=source, idx=int(idx)):
                    return (yield source)[idx]

                case NbOp_Gt_Int64(lhs, rhs):
                    lhs = yield lhs
                    rhs = yield rhs
                    return builder.icmp_signed(">", lhs, rhs)

                case NbOp_Lt_Int64(lhs, rhs):
                    lhs = yield lhs
                    rhs = yield rhs
                    return builder.icmp_signed("<", lhs, rhs)

                case NbOp_Add_Int64(lhs, rhs):
                    lhs = yield lhs
                    rhs = yield rhs
                    return builder.add(lhs, rhs)

                case NbOp_Sub_Int64(lhs, rhs):
                    lhs = yield lhs
                    rhs = yield rhs
                    return builder.sub(lhs, rhs)

                ##### more

                case NbOp_Undef_Int64():
                    return ir.IntType(64)(ir.Undefined)

                case rg.PyBool(val):
                    return ir.IntType(1)(val)

                case rg.PyInt(val):
                    return ir.IntType(64)(val)

                case NbOp_Not_Int64(operand):
                    opval = yield operand
                    return builder.icmp_unsigned("==", opval, opval.type(0))

                case rg.Loop(body=rg.RegionEnd() as body, operands=operands):
                    # process operands
                    ops = []
                    for op in operands:
                        ops.append((yield op))

                    # Note this is a tail loop.
                    begin = body.begin

                    with push(*ops):
                        loopentry_values = yield begin

                    bb_before = builder.basic_block
                    bb_loopbody = builder.append_basic_block("loopbody")
                    bb_endloop = builder.append_basic_block("endloop")
                    builder.branch(bb_loopbody)
                    # loop body
                    builder.position_at_end(bb_loopbody)
                    # setup phi nodes for loopback variables

                    phis = []
                    for i, var in enumerate(loopentry_values):
                        phi = builder.phi(var.type, name=f"loop_{i}")
                        phi.add_incoming(var, bb_before)
                        phis.append(phi)

                    # generate body
                    loop_memo = {begin: tuple(phis)}
                    memo = ase.traverse(
                        body,
                        lower_expr,
                        init_memo=loop_memo,
                    )

                    loopout_values = list(memo[body])
                    # get loop condition
                    loopcond = loopout_values.pop(0)

                    # fix up phis
                    for i, phi in enumerate(phis):
                        assert phi.type == loopout_values[i].type, (
                            phi.type,
                            loopout_values[i].type,
                        )
                        phi.add_incoming(
                            loopout_values[i], builder.basic_block
                        )
                    # back jump
                    builder.cbranch(loopcond, bb_loopbody, bb_endloop)
                    # end loop
                    builder.position_at_end(bb_endloop)
                    # Returns the value from the loop body because this is a tail loop
                    return loopout_values

            raise NotImplementedError(expr)

        try:
            with push(iostate, *fn.args):
                memo = ase.traverse(root.body, lower_expr)
        except:
            print(mod)
            raise

        func_region_outs = memo[root.body]

        i, p = get_port_by_name(root.body.ports, rvsdg.internal_prefix("ret"))
        builder.ret(func_region_outs[i])

        return mod

    def jit_compile(self, llmod: ir.Module, func_node: rg.Func):
        sym = func_node.fname
        # Create JIT
        lljit = llvm.create_lljit_compiler()
        rt = (
            llvm.JITLibraryBuilder()
            .add_ir(str(llmod))
            .export_symbol(sym)
            .add_current_process()
            .link(lljit, sym)
        )
        ptr = rt[sym]

        fnty = llmod.get_global(sym).type.pointee
        ct_args = list(map(self.get_ctype, fnty.args))
        ct_ret = self.get_ctype(fnty.return_type)

        return JitCallable.from_pointer(rt, ptr, ct_args, ct_ret)

    def get_ctype(self, lltype: ir.Type):
        match lltype:
            case ir.IntType():
                match lltype.width:
                    case 64:
                        return ctypes.c_int64
        raise NotImplementedError(lltype)


@dataclass(frozen=True)
class JitCallable:
    rt: llvm.ResourceTracker
    pyfunc: Callable

    @classmethod
    def from_pointer(cls, rt: llvm.ResourceTracker, ptr: int, argtys, retty):
        pyfunc = ctypes.PYFUNCTYPE(retty, *argtys)(ptr)
        return cls(rt=rt, pyfunc=pyfunc)

    def __call__(self, *args: Any) -> Any:
        return self.pyfunc(*args)


def example(a, b):
    if a > b:
        z = a - b
    else:
        z = b - a
    return z - a


Int64 = NbOp_Type("Int64")

if __name__ == "__main__":
    jt = compiler_pipeline(
        example,
        argtypes=(Int64, Int64),
        ruleset=(
            basic_ruleset
            | ruleset_propagate_typeof_ifelse
            | ruleset_type_unify
            | ruleset_type_infer_gt
            | ruleset_type_infer_lt
            | ruleset_type_infer_add
            | ruleset_type_infer_sub
            | ruleset_region_types
            | facts_function_types
        ),
        verbose=True,
        converter_class=ExtendEGraphToRVSDG,
        cost_model=MyCostModel(),
        backend=Backend(),
    )
    res = jt(10, 7)
    run_test(example, jt, (10, 7), verbose=True)
