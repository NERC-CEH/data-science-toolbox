# Notebook Metadata Guidance

This document contains guidance on how to include metadata in your notebook for the NC-UK Data Science Toolbox.

If you are using the [template notebook](https://github.com/NERC-CEH/data-science-toolbox/blob/main/template_notebook/template_notebook.ipynb), the metadata section is pre-filled with placeholder text. You should replace this placeholder text with information that is relevant to your own method. Although the template contains some indications on how to fill-in the metadata section, this document contains more comprehensive guidance.

You do not have to use the template to contribute your notebook to the toolbox, but you must follow the metadata guidance below.

Jupyter Notebooks use [MyST Markdown](https://mystmd.org/) to format frontmatter fields, which are essentially metadata for your notebook. The MyST website has the [full frontmatter documentation](https://mystmd.org/guide/frontmatter), which you may wish to consult if you want to go beyond the following guidance.

## Titles and description

``title``

The full title of your method.

``short_title``

A short title for your method, which must contain fewer than 40 characters. It will be used when there is not enough space for the full title, e.g. in the site navigation panel on the left.

``subtitle``

An optional subtitle for your method. If you include one, it will be displayed below the main title

``description``

A brief description of your method in 1-2 sentences.

## Dates

``date``

The date on which your notebook was first published publicly, in YYYY-MM-DD format.

``updated``

The date on which your notebook was last updated, in YYYY-MM-DD format.

## Licenses

``license``

``code``: the [SPDX Identifier](https://spdx.org/licenses/) of the license for the code in your notebook.

``content``: the [SPDX Identifier](https://spdx.org/licenses/) of the license for the rest of the content in your notebook.

## Links

``github``

The URL of the repository that contains your notebook.

``doi``

The URL of the DOI of your notebook (if you have one).

``thumbnail``

The URL or relative path of an image that is representative of your method. It will be used in link previews for your notebook page.

## Authors

You may include as many authors as required. Expand the "Multiple authors example" section below to see how this can be done.

<details>

<summary>Multiple authors example</summary>

```yaml
authors:  # include this line only once
  - name: "Full Name of Author 1"
    orcid: "0000-0000-0000-0001"
    corresponding: true  # corresponding author, must include email below
    email: "author@example.com"
    affiliations:  # an author may have more than one affiliation
      - name: "Organisation 1 Name"
        ror: "ror-ID-1"
      - name: "Organisation 2 Name"
        ror: "ror-ID-2"
  - name: "Full Name of Author 2"
    orcid: "0000-0000-0000-0002"
    corresponding: false  # not a corresponding author, not necessary to include email
    affiliations:
      - name: "Organisation Name 2"
        ror: "ror-ID-2"
      - name: "Non-Research Organisation"  # does not have a ROR identifier
```

</details>

``name``

The full name of the author.

``orcid``

The author's [ORCID](https://orcid.org/) identifier, without the URL.

``corresponding``

`true` if the author is a corresponding author, `false` otherwise. At least one author must be marked as the corresponding author.

``email``

The email address of the author. If the author is a corresponding author, you must include their email address. If the author is not a corresponding author, including their email address is optional.

### Affiliations

An author may be associated with one or more affiliations.

``name``

The name of the organisation with which the author is affiliated.

``ror``

The organisation's [ROR](https://ror.org/) identifier, without the URL, if it is a research organisation. If it is a research organisation and does not yet have an ROR identifier, you can [suggest](https://docs.google.com/forms/d/e/1FAIpQLSdJYaMTCwS7muuTa-B_CnAtCSkKzt19lkirAKG4u7umH9Nosg/viewform) its addition to ROR. If it is not a research organisation, this field can be omitted.

## Funding

You may include as many funding statements as required. Expand the "Multiple funding statements example" section below to see how this can be done.

<details>

<summary>Multiple funding statements example</summary>

```yaml
funding:  # include this line only once
  - statement: "Funding statement 1."
    awards:  # a funding statement may have more than one associated award
      - id: "award-id-1"
        name: "Award name"
        sources:  # an award may have more than one source (funder)
          - name: "Funder 1 Name"
          - name: "Funder 2 Name"
      - id: "award-id-2"
        name: "Award name"
        sources:
          - name: "Funder 3 Name"
  - statement: "Funding statement 2."
    awards:
      - id: "award-id-3"
        name: "Award name"
        sources:
          - name: "Funder 1 Name"
  - statement: "Funding statement 3."
```

</details>

``statement``

A statement about funding related to your notebook.

### Awards

A statement may be associated with zero, one or more than one awards.

``id``

The ID of an award that the funding statement references.

``name``

The name of the award.

#### Sources

An award may be associated with one or more sources (funders).

``name``

The name of the source that funded the award.
