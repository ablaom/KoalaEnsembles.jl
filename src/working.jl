struct Foo
    X::Vector{Int}
end

object = Foo([7, 7, 7])

object.X

Y = object.X
(object.X)[1] = 8

object.X

Y






mutable struct Foo
    x::Vector{Float64}
end

function bar(object::Foo)

    object.x[1]=2.3
    mu = mean(object.x)

    return object, mu
end

function bar2(object::Foo)
    x = object.x

    x[1]=2.3
    mu = mean(x)

    object.x = x

    return object, mean(x)
end

object = Foo(randn(10^6))

@allocated bar(object)
@allocated bar2(object)


