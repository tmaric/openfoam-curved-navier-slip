/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | foam-extend: Open Source CFD
   \\    /   O peration     |
    \\  /    A nd           | For copyright notice see file Copyright
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of foam-extend.

    foam-extend is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the
    Free Software Foundation, either version 3 of the License, or (at your
    option) any later version.

    foam-extend is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with foam-extend.  If not, see <http://www.gnu.org/licenses/>.

Original Author: 
    Dirk Gründing
Modified By:
    Suraj Raju    

Description:
    Utility to extract the cell center and velocity components
    as a post-processing utility.
\*---------------------------------------------------------------------------*/

#include <tuple>
#include "OFstream.H"
#include "Ostream.H"
#include "fvCFD.H"
#include "fvMesh.H"
#include "Time.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    timeSelector::addOptions();
#   include "setRootCase.H"
#   include "createTime.H"
#   include "createMesh.H"

    mkDir(fileName("evaluation"));

    instantList timeDirs = timeSelector::select0(runTime, args);

    const List<vector> xCenter = mesh.C();
    forAll(timeDirs, timeI)
    {
        word curTimeName(timeDirs[timeI].name());
        Info << "Processing time: " << curTimeName << endl;

        fileName resFileName("evaluation/velocities_" + curTimeName + ".dat");
        OFstream* resStreamPtr(new OFstream(resFileName,IOstream::ASCII,
                                    IOstream::UNCOMPRESSED,IOstream::APPEND ));
        OFstream& resStream(*resStreamPtr);
        resStream 
                << "x"  << tab
                << "y"  << tab
                << "z"  << tab
                << "vx" << tab
                << "vy" << tab
                << "vz"
                << endl;
        runTime.setTime(timeDirs[timeI], timeI);
        const volVectorField U
        (
            IOobject
            (
                "U",
                runTime.timeName(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
            ),
            mesh
        );
        forAll(U.internalField(),cellI)
        {
            resStream 
                << xCenter[cellI].x() << tab
                << xCenter[cellI].y() << tab
                << xCenter[cellI].z() << tab
                << U.internalField()[cellI].x() << tab
                << U.internalField()[cellI].y() << tab
                << U.internalField()[cellI].z()
                << endl;
        }
    }

    Info << "End\n" << endl;

    return(0);
}


// ************************************************************************* //
//
