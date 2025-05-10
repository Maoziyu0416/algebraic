from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Union, List, Optional, Tuple
import math
from collections import defaultdict


class Expression(ABC):
    """抽象基类，定义所有表达式必须实现的方法"""

    @property
    @abstractmethod
    def precedence(self) -> int:
        """返回运算符优先级（数值越大优先级越高）"""
        pass

    @abstractmethod
    def to_str(self, parent_precedence: int = 0) -> str:
        """将表达式转换为字符串形式"""
        pass

    @abstractmethod
    def evaluate(self, **kwargs) -> float:
        """表达式求值"""
        pass

    @abstractmethod
    def derivative(self, variable: str) -> Expression:
        """对指定变量求导"""
        pass

    @abstractmethod
    def coefficients(self) -> Dict[str, float]:
        """获取表达式中各变量的系数"""
        pass

    @abstractmethod
    def collect_terms(self, variable: str) -> Dict[int, float]:
        """收集指定变量的各项系数（幂次:系数）"""
        pass

    # 运算符重载
    def __add__(self, other: Union[Expression, int, float]) -> Add:
        if isinstance(other, (int, float)):
            return Add(self, Constant(other))
        return Add(self, other)

    def __sub__(self, other: Union[Expression, int, float]) -> Subtract:
        if isinstance(other, (int, float)):
            return Subtract(self, Constant(other))
        return Subtract(self, other)

    def __mul__(self, other: Union[Expression, int, float]) -> Multiply:
        if isinstance(other, (int, float)):
            return Multiply(self, Constant(other))
        return Multiply(self, other)

    def __truediv__(self, other: Union[Expression, int, float]) -> Divide:
        if isinstance(other, (int, float)):
            return Divide(self, Constant(other))
        return Divide(self, other)

    def __pow__(self, exponent: Union[Expression, int, float]) -> Power:
        if isinstance(exponent, (int, float)):
            return Power(self, Constant(exponent))
        return Power(self, exponent)

    def __neg__(self) -> Negate:
        return Negate(self)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    # 高级功能
    def factor(self) -> List[Expression]:
        """因式分解"""
        return Factorization.factor(self)

    def solve(self, *variables: str) -> EquationSolver.Solution:
        """方程求解"""
        return EquationSolver.solve_system(self, variables=variables)


@dataclass
class Variable(Expression):
    """变量表达式"""
    name: str

    @property
    def precedence(self) -> int:
        return 5  # 最高优先级

    def evaluate(self, **kwargs) -> float:
        return kwargs[self.name]

    def derivative(self, variable: str) -> Expression:
        return Constant(1) if self.name == variable else Constant(0)

    def coefficients(self) -> Dict[str, float]:
        return {self.name: 1.0}

    def to_str(self, parent_precedence: int = 0) -> str:
        return self.name

    def collect_terms(self, var: str) -> Dict[int, float]:
        """变量项的系数收集"""
        return {1: 1.0} if self.name == var else {0: 0.0}


@dataclass
class Constant(Expression):
    """常数表达式"""
    value: float

    @property
    def precedence(self) -> int:
        return 5  # 最高优先级

    def evaluate(self, **kwargs) -> float:
        return self.value

    def derivative(self, variable: str) -> Expression:
        return Constant(0)

    def coefficients(self) -> Dict[str, float]:
        return {}

    def to_str(self, parent_precedence: int = 0) -> str:
        # 优化浮点数显示
        if isinstance(self.value, float):
            s = f"{self.value:.2f}".rstrip('0').rstrip('.')
            return s if s else "0"
        return str(self.value)

    def collect_terms(self, var: str) -> Dict[int, float]:
        """常数项收集"""
        return {0: self.value}


@dataclass
class Add(Expression):
    """加法表达式"""
    left: Expression
    right: Expression

    @property
    def precedence(self) -> int:
        return 1  # 低优先级

    def evaluate(self, **kwargs) -> float:
        return self.left.evaluate(**kwargs) + self.right.evaluate(**kwargs)

    def derivative(self, variable: str) -> Expression:
        return Add(
            self.left.derivative(variable),
            self.right.derivative(variable)
        )

    def coefficients(self) -> Dict[str, float]:
        return {**self.left.coefficients(), **self.right.coefficients()}

    def to_str(self, parent_precedence: int = 0) -> str:
        s = f"{self.left.to_str(self.precedence)} + {self.right.to_str(self.precedence)}"
        # 根据父级优先级决定是否加括号
        return f"({s})" if parent_precedence > self.precedence else s

    def collect_terms(self, var: str) -> Dict[int, float]:
        """合并同类项"""

        def merge_terms(a: Dict[int, float], b: Dict[int, float]) -> Dict[int, float]:
            result = a.copy()
            for power, coeff in b.items():
                result[power] = result.get(power, 0.0) + coeff
            return result

        return merge_terms(
            self.left.collect_terms(var),
            self.right.collect_terms(var)
        )


@dataclass
class Subtract(Expression):
    """减法表达式"""
    left: Expression
    right: Expression

    @property
    def precedence(self) -> int:
        return 1  # 低优先级

    def evaluate(self, **kwargs) -> float:
        return self.left.evaluate(**kwargs) - self.right.evaluate(**kwargs)

    def derivative(self, variable: str) -> Expression:
        return Subtract(
            self.left.derivative(variable),
            self.right.derivative(variable)
        )

    def coefficients(self) -> Dict[str, float]:
        left_coeffs = self.left.coefficients()
        right_coeffs = self.right.coefficients()
        # 右侧系数取反
        for k, v in right_coeffs.items():
            right_coeffs[k] = -v
        return {**left_coeffs, **right_coeffs}

    def to_str(self, parent_precedence: int = 0) -> str:
        # 右侧优先级+1以避免歧义
        s = f"{self.left.to_str(self.precedence)} - {self.right.to_str(self.precedence + 1)}"
        return f"({s})" if parent_precedence > self.precedence else s

    def collect_terms(self, var: str) -> Dict[int, float]:
        """合并同类项（右侧系数取反）"""
        left_terms = self.left.collect_terms(var)
        right_terms = self.right.collect_terms(var)
        # 右侧系数取反
        for power in right_terms:
            right_terms[power] *= -1
        return {**left_terms, **right_terms}


@dataclass
class Multiply(Expression):
    """乘法表达式"""
    left: Expression
    right: Expression

    @property
    def precedence(self) -> int:
        return 2  # 中等优先级

    def evaluate(self, **kwargs) -> float:
        return self.left.evaluate(**kwargs) * self.right.evaluate(**kwargs)

    def derivative(self, variable: str) -> Expression:
        # 乘积法则：(fg)' = f'g + fg'
        return Add(
            Multiply(self.left.derivative(variable), self.right),
            Multiply(self.left, self.right.derivative(variable))
        )

    def coefficients(self) -> Dict[str, float]:
        left_coeffs = self.left.coefficients()
        right_coeffs = self.right.coefficients()
        # 处理系数相乘关系
        result = {}
        for lk, lv in left_coeffs.items():
            for rk, rv in right_coeffs.items():
                key = f"{lk}*{rk}" if lk and rk else lk or rk
                result[key] = lv * rv
        return result

    def to_str(self, parent_precedence: int = 0) -> str:
        left_str = self.left.to_str(self.precedence)
        right_str = self.right.to_str(self.precedence)
        # 优化显示：常数与变量相乘时省略乘号
        sep = "" if isinstance(self.left, Constant) and isinstance(self.right, Variable) else " * "
        s = f"{left_str}{sep}{right_str}"
        return f"({s})" if parent_precedence > self.precedence else s

    def collect_terms(self, var: str) -> Dict[int, float]:
        """多项式乘法规则"""
        left_terms = self.left.collect_terms(var)
        right_terms = self.right.collect_terms(var)

        result = defaultdict(float)
        for l_pow, l_coeff in left_terms.items():
            for r_pow, r_coeff in right_terms.items():
                result[l_pow + r_pow] += l_coeff * r_coeff
        return dict(result)


@dataclass
class Divide(Expression):
    """除法表达式"""
    numerator: Expression
    denominator: Expression

    @property
    def precedence(self) -> int:
        return 2  # 中等优先级

    def evaluate(self, **kwargs) -> float:
        denom = self.denominator.evaluate(**kwargs)
        if denom == 0:
            raise ZeroDivisionError("Division by zero")
        return self.numerator.evaluate(**kwargs) / denom

    def derivative(self, variable: str) -> Expression:
        # 商法则：(f/g)' = (f'g - fg')/g²
        f = self.numerator
        g = self.denominator
        return Divide(
            Subtract(
                Multiply(f.derivative(variable), g),
                Multiply(f, g.derivative(variable))
            ),
            Power(g, Constant(2))
        )

    def coefficients(self) -> Dict[str, float]:
        return {
            **self.numerator.coefficients(),
            **{f"1/{k}": v for k, v in self.denominator.coefficients().items()}
        }

    def to_str(self, parent_precedence: int = 0) -> str:
        s = f"{self.numerator.to_str(self.precedence)} / {self.denominator.to_str(self.precedence)}"
        return f"({s})" if parent_precedence > self.precedence else s

    def collect_terms(self, var: str) -> Dict[int, float]:
        """分式暂不支持系数收集"""
        raise NotImplementedError("分式表达式暂不支持系数收集")


@dataclass
class Power(Expression):
    """幂运算表达式"""
    base: Expression
    exponent: Expression

    @property
    def precedence(self) -> int:
        return 4  # 高优先级

    def evaluate(self, **kwargs) -> float:
        return self.base.evaluate(**kwargs) ** self.exponent.evaluate(**kwargs)

    def derivative(self, variable: str) -> Expression:
        # 幂法则和链式法则：d/dx [u^v] = u^v * (v' ln(u) + v u' / u)
        return Multiply(
            self,
            Add(
                Multiply(self.exponent.derivative(variable), Ln(self.base)),
                Multiply(self.exponent, Divide(
                    self.base.derivative(variable),
                    self.base
                ))
            )
        )

    def coefficients(self) -> Dict[str, float]:
        return {
            **self.base.coefficients(),
            **self.exponent.coefficients()
        }

    def to_str(self, parent_precedence: int = 0) -> str:
        base_str = self.base.to_str(self.precedence)
        exponent_str = self.exponent.to_str(self.precedence)
        s = f"{base_str}^{exponent_str}"
        return f"({s})" if parent_precedence > self.precedence else s

    def collect_terms(self, var: str) -> Dict[int, float]:
        """仅支持整数幂的系数收集"""
        if isinstance(self.exponent, Constant) and self.exponent.value.is_integer():
            base_terms = self.base.collect_terms(var)
            result = {0: 1.0}  # 初始化为1
            for _ in range(int(self.exponent.value)):
                new_result = defaultdict(float)
                for r_pow, r_coeff in result.items():
                    for b_pow, b_coeff in base_terms.items():
                        new_result[r_pow + b_pow] += r_coeff * b_coeff
                result = dict(new_result)
            return result
        raise NotImplementedError("仅支持整数幂的系数收集")


@dataclass
class Negate(Expression):
    """负号表达式"""
    expr: Expression

    @property
    def precedence(self) -> int:
        return 3  # 较高优先级

    def evaluate(self, **kwargs) -> float:
        return -self.expr.evaluate(**kwargs)

    def derivative(self, variable: str) -> Expression:
        return Negate(self.expr.derivative(variable))

    def coefficients(self) -> Dict[str, float]:
        return {k: -v for k, v in self.expr.coefficients().items()}

    def to_str(self, parent_precedence: int = 0) -> str:
        s = f"-{self.expr.to_str(self.precedence)}"
        return f"({s})" if parent_precedence > self.precedence else s

    def collect_terms(self, var: str) -> Dict[int, float]:
        """负号处理"""
        terms = self.expr.collect_terms(var)
        return {k: -v for k, v in terms.items()}


@dataclass
class Sin(Expression):
    """正弦函数表达式"""
    expr: Expression

    @property
    def precedence(self) -> int:
        return 5  # 最高优先级

    def evaluate(self, **kwargs) -> float:
        return math.sin(self.expr.evaluate(**kwargs))

    def derivative(self, variable: str) -> Expression:
        # 链式法则：sin(u)' = cos(u)*u'
        return Multiply(Cos(self.expr), self.expr.derivative(variable))

    def coefficients(self) -> Dict[str, float]:
        return self.expr.coefficients()

    def to_str(self, parent_precedence: int = 0) -> str:
        return f"sin({self.expr.to_str(0)})"  # 强制参数加括号

    def collect_terms(self, var: str) -> Dict[int, float]:
        """三角函数暂不支持系数收集"""
        raise NotImplementedError("三角函数暂不支持系数收集")


@dataclass
class Cos(Expression):
    """余弦函数表达式"""
    expr: Expression

    @property
    def precedence(self) -> int:
        return 5  # 最高优先级

    def evaluate(self, **kwargs) -> float:
        return math.cos(self.expr.evaluate(**kwargs))

    def derivative(self, variable: str) -> Expression:
        # 链式法则：cos(u)' = -sin(u)*u'
        return Multiply(
            Multiply(Constant(-1), Sin(self.expr)),
            self.expr.derivative(variable)
        )

    def coefficients(self) -> Dict[str, float]:
        return self.expr.coefficients()

    def to_str(self, parent_precedence: int = 0) -> str:
        return f"cos({self.expr.to_str(0)})"  # 强制参数加括号

    def collect_terms(self, var: str) -> Dict[int, float]:
        """三角函数暂不支持系数收集"""
        raise NotImplementedError("三角函数暂不支持系数收集")


@dataclass
class Ln(Expression):
    """自然对数表达式"""
    expr: Expression

    @property
    def precedence(self) -> int:
        return 5  # 最高优先级

    def evaluate(self, **kwargs) -> float:
        val = self.expr.evaluate(**kwargs)
        if val <= 0:
            raise ValueError("Logarithm of non-positive number")
        return math.log(val)

    def derivative(self, variable: str) -> Expression:
        # 导数规则：ln(u)' = u'/u
        return Divide(
            self.expr.derivative(variable),
            self.expr
        )

    def coefficients(self) -> Dict[str, float]:
        return self.expr.coefficients()

    def to_str(self, parent_precedence: int = 0) -> str:
        return f"ln({self.expr.to_str(0)})"  # 强制参数加括号

    def collect_terms(self, var: str) -> Dict[int, float]:
        """对数函数暂不支持系数收集"""
        raise NotImplementedError("对数函数暂不支持系数收集")


class Factorization:
    """因式分解工具类"""

    @staticmethod
    def factor(expr: Expression) -> List[Expression]:
        """因式分解主方法"""
        # 1. 尝试提取公因子
        common = Factorization._find_common_factor(expr)
        if common is not None:
            return common

        # 2. 尝试二次三项式分解
        if isinstance(expr, Add):
            quadratic = Factorization._factor_quadratic(expr)
            if quadratic:
                return quadratic

        # 3. 其他分解策略...

        return [expr]  # 无法分解时返回原表达式

    @staticmethod
    def _find_common_factor(expr: Expression) -> Optional[List[Expression]]:
        """提取最大公因子"""
        if isinstance(expr, Add):
            terms = Factorization._get_add_terms(expr)
            common = terms[0]
            for term in terms[1:]:
                common = Factorization._gcd(common, term)
            if not isinstance(common, Constant) or common.value != 1:
                factored = Multiply(common, Add(*[Divide(t, common) for t in terms]))
                return [common, factored]
        return None

    @staticmethod
    def _get_add_terms(expr: Expression) -> List[Expression]:
        """获取加法表达式的所有项"""
        terms = []
        stack = [expr]
        while stack:
            current = stack.pop()
            if isinstance(current, Add):
                stack.append(current.right)
                stack.append(current.left)
            else:
                terms.append(current)
        return terms

    @staticmethod
    def _gcd(a: Expression, b: Expression) -> Expression:
        """求两个表达式的最大公因子（简化版）"""
        # 实际实现需要更复杂的逻辑
        if isinstance(a, Constant) and isinstance(b, Constant):
            return Constant(math.gcd(int(a.value), int(b.value)))
        return Constant(1)

    @staticmethod
    def _factor_quadratic(expr: Expression) -> Optional[List[Expression]]:
        """二次三项式分解：x² + bx + c = (x + m)(x + n)"""
        # 实现需要先检测是否为标准二次型
        # 简化版实现：
        if isinstance(expr, Add):
            terms = expr.collect_terms("x")  # 假设变量为x
            if set(terms.keys()) == {0, 1, 2}:
                a = terms.get(2, 0)
                b = terms.get(1, 0)
                c = terms.get(0, 0)

                # 求解mn = c/a, m+n = b/a
                # 这里简化处理a=1的情况
                if a == 1:
                    for m in range(-abs(int(c)), abs(int(c)) + 1):
                        if m == 0 and c != 0:
                            continue
                        if c % m == 0:
                            n = int(c / m)
                            if m + n == b:
                                return [
                                    Add(Variable("x"), Constant(m)),
                                    Add(Variable("x"), Constant(n))
                                ]
        return None


class EquationSolver:
    """方程求解器"""

    class Solution:
        """方程解的数据结构"""

        def __init__(self):
            self.vars = {}  # 变量到表达式的映射
            self.free_vars = []  # 自由变量列表

        def __str__(self):
            return "\n".join([f"{var} = {expr.to_str()}" for var, expr in self.vars.items()])

    @classmethod
    def solve_system(cls, *equations: Expression, variables: List[str]) -> Solution:
        """解方程组主方法"""
        solution = cls.Solution()

        # 1. 方程标准化（移项使等式右边为0）
        std_equations = [cls._standardize(eq) for eq in equations]

        # 2. 线性系统检测
        if cls._is_linear_system(std_equations, variables):
            return cls._solve_linear(std_equations, variables)

        # 3. 单变量方程处理
        if len(variables) == 1:
            return cls._solve_single_variable(std_equations[0], variables[0])

        # 4. 其他情况处理
        return solution  # 返回空解

    @staticmethod
    def _standardize(expr: Expression) -> Expression:
        """将方程转换为标准形式：表达式 = 0"""
        if isinstance(expr, Subtract):
            return Add(expr.left, Negate(expr.right))
        return expr

    @classmethod
    def _is_linear_system(cls, equations: List[Expression], variables: List[str]) -> bool:
        """检测是否为线性方程组"""
        for eq in equations:
            if not cls._is_linear(eq, variables):
                return False
        return True

    @classmethod
    def _is_linear(cls, expr: Expression, variables: List[str]) -> bool:
        """检测表达式对于指定变量是否为线性"""
        if isinstance(expr, Variable):
            return expr.name in variables
        if isinstance(expr, Constant):
            return True
        if isinstance(expr, (Add, Subtract)):
            return cls._is_linear(expr.left, variables) and cls._is_linear(expr.right, variables)
        if isinstance(expr, Multiply):
            # 允许常数乘以线性表达式
            left_linear = cls._is_linear(expr.left, variables)
            right_linear = cls._is_linear(expr.right, variables)
            return (left_linear and isinstance(expr.right, Constant)) or \
                (right_linear and isinstance(expr.left, Constant))
        return False

    @classmethod
    def _solve_linear(cls, equations: List[Expression], variables: List[str]) -> Solution:
        """高斯消元法解线性方程组"""
        solution = cls.Solution()

        # 1. 构建增广矩阵
        matrix = []
        for eq in equations:
            row = [cls._get_coefficient(eq, var) for var in variables]
            row.append(-cls._get_constant_term(eq))
            matrix.append(row)

        # 2. 高斯消元
        n = len(variables)
        for col in range(n):
            # 寻找主元行
            pivot_row = None
            for row in range(col, len(matrix)):
                if matrix[row][col] != 0:
                    pivot_row = row
                    break

            if pivot_row is None:
                solution.free_vars.append(variables[col])
                continue

            # 交换行
            matrix[col], matrix[pivot_row] = matrix[pivot_row], matrix[col]

            # 归一化
            pivot_val = matrix[col][col]
            for c in range(col, n + 1):
                matrix[col][c] /= pivot_val

            # 消元
            for row in range(len(matrix)):
                if row != col and matrix[row][col] != 0:
                    factor = matrix[row][col]
                    for c in range(col, n + 1):
                        matrix[row][c] -= factor * matrix[col][c]

        # 3. 解析结果
        for row in range(len(matrix)):
            all_zero = all(abs(matrix[row][c]) < 1e-10 for c in range(n))
            if all_zero and abs(matrix[row][n]) > 1e-10:
                raise ValueError("无解")

        for col in range(n):
            if variables[col] in solution.free_vars:
                continue

            # 构造解的表达式
            expr = Constant(matrix[col][n])
            for fv in solution.free_vars:
                if fv in variables[col + 1:]:
                    fv_col = variables.index(fv)
                    coeff = -matrix[col][fv_col]
                    if abs(coeff) > 1e-10:
                        expr = Add(expr, Multiply(Constant(coeff), Variable(fv)))

            solution.vars[variables[col]] = expr

        # 添加自由变量
        for fv in solution.free_vars:
            solution.vars[fv] = Variable(fv)

        return solution

    @staticmethod
    def _get_coefficient(expr: Expression, var: str) -> float:
        """获取表达式中指定变量的系数"""
        terms = expr.collect_terms(var)
        return terms.get(1, 0.0)  # 线性项系数

    @staticmethod
    def _get_constant_term(expr: Expression) -> float:
        """获取表达式中的常数项"""
        # 假设变量为x，收集x^0的系数
        terms = expr.collect_terms("x")  # 这里的变量名不影响结果
        return terms.get(0, 0.0)

    @classmethod
    def _solve_single_variable(cls, equation: Expression, variable: str) -> Solution:
        """解单变量方程"""
        solution = cls.Solution()
        terms = equation.collect_terms(variable)

        if not terms:
            solution.vars[variable] = Constant(0)
            return solution

        # 线性方程 ax + b = 0 → x = -b/a
        if 1 in terms and 0 in terms:
            a = terms[1]
            b = terms[0]
            solution.vars[variable] = Divide(Constant(-b), Constant(a))
        elif 1 in terms:
            a = terms[1]
            solution.vars[variable] = Constant(0)
        elif 0 in terms:
            if abs(terms[0]) < 1e-10:
                solution.vars[variable] = Variable(variable)  # 任意解
            else:
                raise ValueError("无解")
        else:
            raise NotImplementedError("仅支持线性方程")

        return solution


# 测试代码
if __name__ == "__main__":
    # 基本表达式测试
    x, y = Variable("x"), Variable("y")

    # 测试表达式构建
    expr1 = 2 * x + 3 * y - 5
    print(f"表达式: {expr1.to_str()}")
    print(f"系数: {expr1.coefficients()}")
    print(f"x的系数: {expr1.collect_terms('x')}")
    print(f"在x=1,y=2处的值: {expr1.evaluate(x=1, y=2)}")

    # 测试导数
    df_dx = expr1.derivative("x")
    print(f"对x求导: {df_dx.to_str()}")

    # 测试因式分解
    expr2 = x ** 2 + 3 * x + 2
    factors = expr2.factor()
    print(f"因式分解: {' * '.join(f.to_str() for f in factors)}")

    # 测试方程求解
    eq1 = 2 * x + 3 * y - 6
    eq2 = x - y + 2
    solution = EquationSolver.solve_system(eq1, eq2, variables=["x", "y"])
    print("方程解:")
    print(solution)

    # 测试单变量方程
    eq3 = x ** 2 - 4  # 注意：当前仅支持线性方程
    try:
        solution = eq3.solve("x")
        print(solution)
    except NotImplementedError as e:
        print(f"错误: {e}")