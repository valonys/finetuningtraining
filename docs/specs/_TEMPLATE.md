# Spec: <feature name>

## Goal
One sentence describing what this feature does and why it matters.

## Non-goals
What this feature explicitly does NOT do (scope guard).

## API surface
New or changed endpoints, functions, or components.

```
POST /v1/example       New endpoint
app/module/file.py     New module
Component.tsx          New/changed UI component
```

## Data flow
How data moves through the system for this feature.

## Acceptance criteria
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] All existing tests still pass
- [ ] New tests cover the happy path and at least one error case

## Test plan
- **Unit:** What to mock, what to test
- **Integration:** Which notebook, curl sequence, or UI flow validates it

## Estimated effort
X days.

## Dependencies
List any features or infra that must exist first.
