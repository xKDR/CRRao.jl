sanction = dataset("Zelig", "sanction")

model = @fitmodel((Num ~ Target + Coop + NCost), sanction, GeometricRegression())
@test sizeof(model) > 0