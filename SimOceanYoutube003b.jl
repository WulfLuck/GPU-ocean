# SimOceanYoutube003b.jl

#   https://www.youtube.com/watch?v=ge0b00gCv5k JuliaCon2022  Simone Silvestri  1h03

#   https://github.com/glwagner/JuliaCon2022-Oceananigans/blob/main/GPU-ocean/visualize.jl

import Pkg; 

#Pkg.add(url="https://github.com/CliMA/ClimaOcean.jl.git"); 

Pkg.add(url="https://github.com/CliMA/ClimaOcean.jl");          Pkg.instantiate      # Pkg.add("CairoMakie") 
#Pkg.add("Oceananigans")
#Pkg.add(url="https://github.com/CliMA/Oceananigans.jl.git");    Pkg.instantiate
#Pkg.add(url="https://github.com/CliMA/Oceananigans.jl"); Pkg.instantiate
Pkg.add("Oceananigans")
Pkg.add("GLMakie"); Pkg.add("JLD2"); Pkg.add("CUDA")

using Oceananigans
using GLMakie
using JLD2
using CUDA

arch = GPU()

Nx = 128
Ny = 60
Nz = 12
Lz = 3600
σ = 1.15
Δz(k) = 3600 * (σ - 1) * σ^(Nz - k) / (σ^Nz - 1)
z_faces(k) = k == 1 ? -Lz : -Lz + sum(Δz.(1:k-1))

#underlying_grid = LatitudeLongitudeGrid(arch, size = (Nx, Ny, Nz), longitude = (-180, 180), latitude = (-84.375, 84.375), z=z_faces)
underlying_grid = LatitudeLongitudeGrid(arch, size = (Nx, Ny, Nz),
                                              latitude = (-84.375, 84.375), 
                                              longitude = (-180, 180),
                                              z = z_faces)
bathymetry = jldopen("bathymetry_juliacon.jld2")["bathymetry"]
#surface(bathymetry * 0.001)
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bathymetry))




#### Physics!

coriolis = HydrostaticSphericalCoriolis()

buoyancy = SeawaterBuoyancy()

### Diffusivity

horizontal_diffusivity = HorizontalScalarBiharmonicDiffusivity(ν = 1e+5, κ = 1e+2)
vertical_diffusivity = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν = 1, κ = 1e-3)

# https://clima.github.io/OceananigansDocumentation/stable/model_setup/turbulent_diffusivity_closures_and_les_models/#Convective-Adjustment-Vertical-Diffusivity%E2%80%93Viscosity
convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0)

closure = (horizontal_diffusivity, vertical_diffusivity, convective_adjustment)

### Boundary conditions (u, v, T, S)

# video 1.25
file_boundary_conditions = jldopen("boundary_conditions_juliacon.jld2")

τx = file_boundary_conditions["τˣ"]
τy = file_boundary_conditions["τʸ"]
Ts = file_boundary_conditions["surface_T"]

heatmap(τx)
heatmap(τy)
heatmap(Ts)

# ValueBoundaryCondition (Dirichlet), GradientBoundaryCondition (Neumann), FluxBoundaryCondition

# check potential speedup   respectively MIT verion
@inline function arch_array(arch, CPUArray)
    GPUArray = CuArray(CPUArray)
    return GPUArray
end



τx = arch_array(arch, τx)
τy = arch_array(arch, τy)
Ts = arch_array(arch, Ts)


#=

u_top_bc = FluxBoundaryCondition(τx)
v_top_bc = FluxBoundaryCondition(τy)

@inline function restoring_T(i, j, grid, clock, fields, parameters)

    T_surface = fields.T[i, j, grid.Nz]
    T_target = parameters.T[i, j]

    # lambda restoring parameter
    flux = parameters.λ * (T_surface - T_target)

    return flux 
end

# video 1:33
#T_top_bc = FluxBoundaryCondition(restoring_T, discrete_form = true, parameters = (Ts = Ts, λ = 0.001))

# https://clima.github.io/OceananigansDocumentation/stable/model_setup/boundary_conditions/
u_bcs = FieldBoundaryConditions(top = u_top_bc)
v_bcs = FieldBoundaryConditions(top = v_top_bc)

# from sim_ocean001    >
#u_bottom_bc = FluxBoundaryCondition(u_linear_drag, discrete_form=true, parameters = 0.01)
#v_bottom_bc = FluxBoundaryCondition(v_linear_drag, discrete_form=true, parameters = 0.01) 
#u_bcs = FieldBoundaryConditions(top = u_top_bc, bottom = u_bottom_bc); v_bcs = FieldBoundaryConditions(top = v_top_bc, bottom = v_bottom_bc);

T_top_bc = FluxBoundaryCondition(restoring_T, discrete_form = true, parameters = (Ts = Ts, λ = 0.001))
T_bcs = FieldBoundaryConditions(top = T_top_bc) #<


#T_top_bc = FieldBoundaryConditions(top = T_top_bc)



#boundary_conditions = (u = u_bcs, v = v_bcs, T = T_top_bc)

##### Model!
model = HydrostaticFreeSurfaceModel(;   grid,
                                        coriolis,
                                        #free_surface = ImplicitFreeSurface(), #<- from sim_ocean001.jl
                                        buoyancy,
                                        tracers = (:T, :S),
                                        closure,
                                        #boundary_conditions
                                        boundary_conditions = (u = u_bcs, v = v_bcs, T = T_bcs)) # sim_ocean001        
     

=#

#=

in sim_ocean001.jl  this model initialization ran:
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bathymetry)); coriolis = HydrostaticSphericalCoriolis(); buoyancy = SeawaterBuoyancy(); closure = (horizontal_diffusivity, vertical_diffusivity, convective_adjustment)
u_bcs = FieldBoundaryConditions(top = u_top_bc, bottom = u_bottom_bc); v_bcs = FieldBoundaryConditions(top = v_top_bc, bottom = v_bottom_bc); T_bcs = FieldBoundaryConditions(top = T_top_bc)

model = HydrostaticFreeSurfaceModel(; grid,
                                      coriolis,
                                      free_surface = ImplicitFreeSurface(),
                                      buoyancy,
                                      tracers = (:T, :S),
                                      #tracer_advection = WENO5(grid),
                                      closure = (vertical_diffusion, horizontal_diffusion, convective_adjustment),
                                      boundary_conditions = (u = u_bcs, v = v_bcs, T = T_bcs))
                                      


=#


#=
Nx = 128
Ny = 60
Nz = 12


Lz = 3600

σ = 1.15
Δz(k) = Lz * (σ - 1) * σ^(Nz - k) / σ^(Nz - 1)                  # Lz  !
z_faces(k) = k == 1 ? - Lz : -Lz + sum(Δz.(1:k-1))



=#

# CUDA.CuArray arch_array(arch, CPUArray) = CuArray(CPUArray)