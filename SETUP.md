# E231project — setup

Everything you need is your own Google Cloud project (with billing) and a Google account for Colab and Drive. No project ID, bucket name or credential is stored in this repository. Read README.md first; SPEC.md has the design (Section 7 lists the notebooks) and CLAUDE.md has the project rules.

## 1. Get the code into Google Drive
Clone or download this repository (GitHub → Code → Download ZIP), unzip it, and upload the folder to Google Drive as `MyDrive/E231project`. That's the notebooks' default `DRIVE_PROJECT_DIR`. A folder nested one level deeper is also found automatically.

## 2. Project ID and bucket
You don't need to edit any files. In the notebook's **Settings** cell, set:
- `PROJECT_ID`: from the project dropdown in the Google Cloud console.
- `BUCKET`: a new bucket name of your choice (lowercase; names are global, so pick something unique).

§2 of the notebook copies the project to `/content/E231project` and writes both values into that working copy, replacing the placeholders `YOUR_GCP_PROJECT_ID` and `YOUR_BUCKET` in the configs. Your Drive folder is not modified. §3 creates the bucket if it doesn't exist. Sign-in happens in Colab (`auth.authenticate_user()`) with your own Google account; nothing is stored in the repository.

To read a teammate's existing data instead of rebuilding it, they can grant "Storage Object Viewer" on their bucket to your account and to your Dataproc service account (`<project-number>-compute@developer.gserviceaccount.com`). Writing results still needs your own bucket.

## 3. One-time Google Cloud setup (done by §3 of either notebook)
§3 does these steps:
- enables the Dataproc, Compute Engine, Cloud Storage and Cloud Resource Manager APIs
- gives the default compute service account the Dataproc Worker and Storage Object Admin roles
- creates the bucket in `infra.region` with the label `project=da231o`
- prints **every quota** a cluster uses, and the largest worker count that fits

Do these yourself in the console:
- **Set a budget alert**, with promotional credits excluded, so it tracks real usage.
- **For 16-worker runs:** upgrade from the Free Trial, then request more quota under IAM & Admin → Quotas & System Limits. The run needs about 68 vCPUs across regional CPUs, N2 CPUs and CPUs (all regions), plus about 1,700 GB of SSD and enough in-use IP addresses.

## 4. Streaming files (done by §6, one time)
The Dataproc clusters have no internet access. So §6 downloads Kafka and Spark 3.5.3's Kafka connector in Colab, which does have internet, and copies them into the bucket. Files already there are skipped.

## 5. Order
Run the cells top to bottom. With `SKIP_DONE = True`, a rerun skips a stage only if its outputs exist **and** were produced with the same settings (a small manifest is written next to the results when each stage finishes). Change `config.yaml` (for example the categories) and the affected stages run again automatically.

**Cost:** you pay for the clusters while they run (each notebook prints the cluster shape before creating it). A first test with `E231_Project_Test.ipynb` on a few small categories and 2–4 workers is cheap; the full 16-worker run takes several hours.

**Always run the last cell (delete the cluster)**, including after a failure.
