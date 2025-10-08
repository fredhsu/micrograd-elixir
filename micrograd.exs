defmodule Value do
  defstruct data: 0, prev: [], grad: 0.0, op: :none, label: ""

  def add(a, b) do
    %Value{data: a.data + b.data, prev: [a, b], op: :add}
  end

  def mul(a, b) do
    %Value{data: a.data * b.data, prev: [a, b], op: :mul}
  end

  def pow(a, b) do
    %Value{data: :math.pow(a.data, b.data), prev: [a, b], op: :pow}
  end

  def backward(%Value{prev: [a, b], op: :add} = val) do
    a =
      %{a | grad: a.grad + val.grad}
      |> backward

    b =
      %{b | grad: b.grad + val.grad}
      |> backward

    %{val | prev: [a, b]}
  end

  def backward(%Value{prev: [a, b], op: :mul} = val) do
    a =
      %{a | grad: a.grad + b.data * val.grad}
      |> backward

    b =
      %{b | grad: b.grad + a.data * val.grad}
      |> backward

    %{val | prev: [a, b]}
  end

  def backward(%Value{prev: [a, b], op: :pow} = val) do
    %{a | grad: a.grad + b.data * :math.pow(a.data, b.data - 1) * val.grad}
    |> backward
  end

  def backward(val) do
    val
  end
end

defmodule Test do
  # L=f((a*b) + c)
  # a=2.0
  # b=-3.0
  # c=10.0
  # f=-2.0
  # Then L = -8
  # a*b=e
  # e+c=d
  # f.grad=4
  # d.grad=-2
  # e.grad=-2
  # c.grad=-2
  # a.grad=6
  # b.grad=-4
  def test do
    a = %Value{data: 2, label: "a"}
    b = %Value{data: -3, label: "b"}
    c = %Value{data: 10, label: "c"}
    e = %{Value.mul(a, b) | label: "e"}
    d = %{Value.add(e, c) | label: "d"}
    f = %Value{data: -2.0, label: "f"}
    g = %Value{data: 3.0, label: "g"}

    IO.inspect(Value.pow(a, g))
    %{Value.mul(f, d) | label: "L"}
  end

  def doutput do
    o =
      %{test() | grad: 1.0}
      |> Value.backward()

    # dode = Value.backward(o)
    # Enum.map(dode.prev, &Value.backward(&1))
  end
end
