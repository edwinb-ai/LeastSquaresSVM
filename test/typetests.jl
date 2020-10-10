@testset "Types" begin
    svm = LSSVC(3, 3)
    @test svm.x == zeros(3, 3)
    @test svm.y == zeros(3)
    @test svm.α == zeros(3)
    @test svm.b == 0.0
end
