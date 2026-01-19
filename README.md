# YML
- ERC 프로젝트 코드 (ML/전처리/예측 포함)

## 스냅샷(결과/데이터 포함) 공유 방법: GitHub Release + ZIP 첨부
이 저장소는 **코드 중심**으로 관리하고, 데이터/결과/문서 등 “전체 폴더 스냅샷”은 **Release에 ZIP으로 첨부**하는 방식을 권장합니다.

### 1) 스냅샷 ZIP 만들기 (로컬)
PowerShell에서 아래 실행:

```powershell
cd "C:\Users\thf56\Documents\YML"
powershell -ExecutionPolicy Bypass -File .\scripts\make_snapshot.ps1
```

생성물: `release_assets\YML-snapshot-YYYYMMDD-HHMMSS.zip`

### 2) GitHub Release 만들고 ZIP 첨부
GitHub 저장소에서 **Releases → Draft a new release**로 들어가서,

- Tag(예: `snapshot-20260119-1`)
- Title(예: `Snapshot 2026-01-19`)
- Assets에 위 ZIP 파일 첨부

후 Publish 하면 됩니다.