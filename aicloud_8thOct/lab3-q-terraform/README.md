# CI/CD Artifact Bucket

We need an AWS S3 bucket for storing production CI/CD build artifacts.

## Requirements

- The bucket is private.
- Public access must be blocked.
- S3 versioning must be enabled.
- Server-side encryption must be enabled.
- The bucket must have Environment and Owner tags.
- The design should avoid unnecessarily permissive IAM permissions.
- The Terraform should be reusable and easy to understand.
- Use Terraform 1.x syntax.

The infrastructure should be suitable as a starting point for a production environment.

> **Note**: Do not assume that generated Terraform is automatically production-ready. Human review and validation are required.