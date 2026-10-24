"""The six reported latent-geometry statistics and the radial amplitude profile."""
import argparse
import json
from pathlib import Path
import numpy as np
from . import latent_bases as lb


def analyse(latents, pixel_angles=None):
    """Input: raw encoder latents (clips, frames, height, width, channels).

    Use the globally scaled encoder latents for the paper geometry table; the
    per-coordinate scale used by a DINO world model changes the covariance.
    """
    if latents.ndim != 5 or latents.shape[2] != latents.shape[3]:
        raise ValueError('Expected (clips, frames, side, side, channels)')
    n,t,h,w,c=latents.shape
    frames=np.ascontiguousarray(latents,dtype=np.float32).reshape(n*t,-1)
    mean=frames.mean(0,dtype=np.float64)
    ev = (lb.gram_eigenvalues(frames,mean) if frames.shape[1]>frames.shape[0]
          else np.linalg.eigvalsh(lb.covariance(frames,mean))[::-1].clip(0))
    result=lb.spectrum_scalars(ev)
    angles=lb.turning_angles(frames.reshape(n,t,-1))
    result['trajectory_theta_mean_latent_rad']=float(angles.mean())
    if pixel_angles is not None:
        if np.asarray(pixel_angles).shape != (n,t-2):
            raise ValueError('Pixel angles must have shape (clips, frames - 2)')
        result['trajectory_theta_mean_pixel_rad']=float(np.mean(pixel_angles))
        result['trajectory_rho']=float(angles.mean()/(np.mean(pixel_angles)+1e-12))
    else:
        result['trajectory_rho']=None
    result.update(lb.within_across_distance(frames,np.repeat(np.arange(n),t)))
    maps=frames.reshape(-1,h,w,c)
    spectrum=lb.amplitude_spectrum(maps)
    profile=lb.radial_profile(spectrum)
    k=np.arange(1,len(profile)+1)
    result['spectral_slope']=float(np.polyfit(np.log(k),np.log(profile),1)[0])
    power=profile**2
    result['hf_power_share']=float(power[k>=8*len(profile)//16].sum()/power.sum())
    result.update(n_frames=n*t,n_clips=n,latent_shape=[h,w,c])
    return result,dict(radial_profile=profile)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--latents',type=Path,required=True)
    ap.add_argument('--pixel-angles',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    result,arrays=analyse(np.load(args.latents,mmap_mode='r'),
                          np.load(args.pixel_angles) if args.pixel_angles else None)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'structure.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(args.output/'structure_arrays.npz',**arrays)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
