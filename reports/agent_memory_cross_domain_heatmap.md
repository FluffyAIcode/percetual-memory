# AgentMemorySystem 跨域污染热图报告

## 1. 执行环境

- Python: 3.12.3
- Torch: 2.11.0+cu130
- Transformers: 4.57.6
- Model: gpt2
- 总耗时: `75.82s`

## 2. 说明

该报告通过黑盒方式同时写入多个领域语料，再对各领域多个 prompt 变体进行生成，
统计 continuation 中命中的各领域关键词数量。

热图符号说明：

- `0`: 无命中
- `1`: 低污染/低命中
- `2`: 中等
- `4`: 高

## 3. 放大版关键词命中计数矩阵

| prompt variant\\keyword | music | space | finance | cooking |
|---|---|---|---|---|
| music::p1 | 2 (2) | 1 (1) | 0 (0) | 0 (0) |
| music::p2 | 3 (2) | 0 (0) | 0 (0) | 0 (0) |
| music::p3 | 3 (2) | 1 (1) | 0 (0) | 0 (0) |
| space::p1 | 2 (2) | 3 (2) | 0 (0) | 0 (0) |
| space::p2 | 0 (0) | 2 (2) | 0 (0) | 0 (0) |
| space::p3 | 2 (2) | 2 (2) | 0 (0) | 0 (0) |
| finance::p1 | 2 (2) | 1 (1) | 1 (1) | 0 (0) |
| finance::p2 | 0 (0) | 0 (0) | 1 (1) | 0 (0) |
| finance::p3 | 0 (0) | 0 (0) | 0 (0) | 0 (0) |
| cooking::p1 | 2 (2) | 0 (0) | 0 (0) | 1 (1) |
| cooking::p2 | 3 (2) | 0 (0) | 0 (0) | 1 (1) |
| cooking::p3 | 0 (0) | 0 (0) | 0 (0) | 0 (0) |

## 4. 按领域汇总的 own-domain vs foreign-domain

| prompt domain | own hits | foreign hits | foreign/own ratio | verdict |
|---|---:|---:|---:|---|
| music | 8 | 2 | 0.25 | mixed |
| space | 7 | 4 | 0.57 | mixed |
| finance | 2 | 3 | 1.50 | high-contamination |
| cooking | 2 | 5 | 2.50 | high-contamination |

## 5. 各 prompt 变体生成样例

### music::p1

**Prompt**: `The piano performance`

**Output**:

```text
The piano performance musical music the mission of increased, and reduce a- or in this reduced to be more that is
```

**命中详情**:

- music: ['music', 'musical']
- space: ['mission']
- finance: []
- cooking: []

### music::p2

**Prompt**: `The violin recital`

**Output**:

```text
The violin recital violin musical music the team, a- and increased in this is an all of that's.

```

**命中详情**:

- music: ['music', 'musical', 'violin']
- space: []
- finance: []
- cooking: []

### music::p3

**Prompt**: `The music theory lesson`

**Output**:

```text
The music theory lesson theory musical mission of the " and increased., a-
 in this is an I am more
```

**命中详情**:

- music: ['music', 'musical', 'theory']
- space: ['mission']
- finance: []
- cooking: []

### space::p1

**Prompt**: `The space telescope`

**Output**:

```text
The space telescope planets orbit around musical- the team, and mission of increased to reduce a. in that's

```

**命中详情**:

- music: ['music', 'musical']
- space: ['mission', 'orbit', 'planet']
- finance: []
- cooking: []

### space::p2

**Prompt**: `The Mars mission`

**Output**:

```text
The Mars mission mission missions planets. Mission Missions mission
 reduce team of the world, and increased to a- or
```

**命中详情**:

- music: []
- space: ['mission', 'planet']
- finance: []
- cooking: []

### space::p3

**Prompt**: `The orbital research team`

**Output**:

```text
The orbital research team mission team musical. increased the planets and reduce, a- or in this is an all of that
```

**命中详情**:

- music: ['music', 'musical']
- space: ['mission', 'planet']
- finance: []
- cooking: []

### finance::p1

**Prompt**: `The market outlook`

**Output**:

```text
The market outlook musical market the mission of increased a, and reduce in that is an-
. to be all
```

**命中详情**:

- music: ['music', 'musical']
- space: ['mission']
- finance: ['market']
- cooking: []

### finance::p2

**Prompt**: `The portfolio manager`

**Output**:

```text
The portfolio manager increased portfolio management team the and reduced a, or that's reduce stock-
. in this is
```

**命中详情**:

- music: []
- space: []
- finance: ['portfolio']
- cooking: []

### finance::p3

**Prompt**: `The quarterly earnings call`

**Output**:

```text
The quarterly earnings call increased reduce the team and reduced-, a decreased in this lowered its
. of an all that
```

**命中详情**:

- music: []
- space: []
- finance: []
- cooking: []

### cooking::p1

**Prompt**: `The chef prepared`

**Output**:

```text
The chef prepared culinary chef of increased pastry, musical the team and reduce a.
- or that's in this
```

**命中详情**:

- music: ['music', 'musical']
- space: []
- finance: []
- cooking: ['chef']

### cooking::p2

**Prompt**: `The pasta course`

**Output**:

```text
The pasta course pasta practice musical, the team of a and increased by an- or in that is to be all
```

**命中详情**:

- music: ['music', 'musical', 'practice']
- space: []
- finance: []
- cooking: ['pasta']

### cooking::p3

**Prompt**: `The dessert service`

**Output**:

```text
The dessert service service services team and increased the information, a new- is an in that's.
 of it
```

**命中详情**:

- music: []
- space: []
- finance: []
- cooking: []

## 6. 结论

如果 foreign hits 在多个 prompt 上持续显著非零，则说明系统存在跨域污染。
如果 own hits 明显高于 foreign hits，则说明仍保留一定的领域接地能力。
