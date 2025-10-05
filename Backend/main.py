from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="NASA App Backend", version="1.0.0")

# Configure CORS to allow requests from Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the data model for incoming requests
class PlanetData(BaseModel):
    # Numeric fields
    sy_snum: Optional[float] = 0
    sy_pnum: Optional[float] = 0
    disc_year: Optional[float] = 0
    pl_orbper: Optional[float] = 0
    pl_orbsmax: Optional[float] = 0
    pl_orbeccen: Optional[float] = 0
    pl_rade: Optional[float] = 0
    pl_radj: Optional[float] = 0
    pl_bmasse: Optional[float] = 0
    pl_bmassj: Optional[float] = 0
    pl_insol: Optional[float] = 0
    pl_eqt: Optional[float] = 0
    st_teff: Optional[float] = 0
    st_rad: Optional[float] = 0
    ttv_flag: Optional[float] = 0
    st_mass: Optional[float] = 0
    st_met: Optional[float] = 0
    st_logg: Optional[float] = 0
    ra: Optional[float] = 0
    dec: Optional[float] = 0
    sy_dist: Optional[float] = 0
    sy_vmag: Optional[float] = 0
    sy_kmag: Optional[float] = 0
    sy_gaiamag: Optional[float] = 0
    planet_star_radius_ratio: Optional[float] = 0
    planet_star_mass_ratio: Optional[float] = 0
    
    # Text fields (not used in sum calculation)
    pl_name: Optional[str] = ""
    hostname: Optional[str] = ""
    discoverymethod: Optional[str] = ""
    disc_facility: Optional[str] = ""
    soltype: Optional[str] = ""
    pl_bmassprov: Optional[str] = ""
    st_spectype: Optional[str] = ""
    st_metratio: Optional[str] = ""

class SumResponse(BaseModel):
    sum: float
    message: str
    field_count: int

@app.get("/")
async def root():
    return {"message": "NASA App Backend API", "status": "running"}

@app.post("/api/calculate-sum", response_model=SumResponse)
async def calculate_sum(data: PlanetData):
    """
    Calculate the sum of all numeric fields from the planet data
    """
    try:
        # Extract all numeric fields and calculate sum
        numeric_values = [
            data.sy_snum or 0,
            data.sy_pnum or 0,
            data.disc_year or 0,
            data.pl_orbper or 0,
            data.pl_orbsmax or 0,
            data.pl_orbeccen or 0,
            data.pl_rade or 0,
            data.pl_radj or 0,
            data.pl_bmasse or 0,
            data.pl_bmassj or 0,
            data.pl_insol or 0,
            data.pl_eqt or 0,
            data.st_teff or 0,
            data.st_rad or 0,
            data.ttv_flag or 0,
            data.st_mass or 0,
            data.st_met or 0,
            data.st_logg or 0,
            data.ra or 0,
            data.dec or 0,
            data.sy_dist or 0,
            data.sy_vmag or 0,
            data.sy_kmag or 0,
            data.sy_gaiamag or 0,
            data.planet_star_radius_ratio or 0,
            data.planet_star_mass_ratio or 0,
        ]
        
        total_sum = sum(numeric_values)
        non_zero_count = sum(1 for v in numeric_values if v != 0)
        
        return SumResponse(
            sum=total_sum,
            message=f"Successfully calculated sum of {non_zero_count} non-zero fields",
            field_count=non_zero_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating sum: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)